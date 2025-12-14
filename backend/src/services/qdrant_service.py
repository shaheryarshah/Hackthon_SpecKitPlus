from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
from .. import config
import logging

logger = logging.getLogger(__name__)

class QdrantService:
    """
    Service class to handle all Qdrant vector database operations
    """

    def __init__(self):
        # Initialize the _initialized flag first
        self._initialized = False
        try:
            self.client = QdrantClient(
                url=config.settings.QDRANT_URL,
                api_key=config.settings.QDRANT_API_KEY,
                prefer_grpc=True  # Use gRPC for better performance if available
            )
            self.collection_name = "book_chunks"
            self._initialize_collection()
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize QdrantService: {e}")
            # Don't raise the exception, just keep _initialized as False
    
    def _initialize_collection(self):
        """
        Initialize the collection with the required vector configuration
        """
        if not self._initialized:
            logger.warning("QdrantService not initialized, skipping collection initialization")
            return

        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(coll.name == self.collection_name for coll in collections.collections)

            if not collection_exists:
                # Create the collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # Default OpenAI embedding size
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            raise
    
    def upsert_vectors(self, points: List[Dict[str, Any]]):
        """
        Upsert vectors into the collection
        
        Args:
            points: List of dictionaries with 'id', 'vector', and 'payload' keys
        """
        try:
            # Prepare points in the required format
            qdrant_points = []
            for point in points:
                qdrant_points.append(
                    models.PointStruct(
                        id=point["id"],
                        vector=point["vector"],
                        payload=point["payload"]
                    )
                )
            
            # Upsert the points
            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points
            )
            
            logger.info(f"Upserted {len(points)} vectors to collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            raise
    
    def search_vectors(self,
                      query_vector: List[float],
                      top_k: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection

        Args:
            query_vector: The vector to search for similarity
            top_k: Number of results to return
            filters: Optional filters to apply to the search

        Returns:
            List of matching points with payload data
        """
        if not self._initialized:
            logger.warning("QdrantService not initialized, returning empty results")
            return []  # Return empty list instead of raising an exception

        try:
            # Build filter conditions if provided
            search_filter = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=str(value))
                        )
                    )

                if filter_conditions:
                    search_filter = models.Filter(
                        must=filter_conditions
                    )

            # Perform the search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })

            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []  # Return empty list instead of raising an exception
    
    def delete_by_payload(self, key: str, value: Any):
        """
        Delete points that match a specific payload condition
        
        Args:
            key: The payload key to match
            value: The payload value to match
        """
        try:
            # Create filter for deletion
            delete_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=str(value))
                    )
                ]
            )
            
            # Delete matching points
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=delete_filter
                )
            )
            
            logger.info(f"Deleted points with {key}={value}, operation ID: {operation_info.operation_id}")
        except Exception as e:
            logger.error(f"Error deleting vectors by payload: {e}")
            raise
    
    def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific point by ID

        Args:
            point_id: The ID of the point to retrieve

        Returns:
            Point data if found, None otherwise
        """
        if not self._initialized:
            logger.warning("QdrantService not initialized, returning None")
            return None

        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )

            if points:
                point = points[0]
                return {
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector
                }

            return None
        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {e}")
            return None
    
    def get_all_chunk_ids(self) -> List[str]:
        """
        Get all stored chunk IDs from the collection
        
        Returns:
            List of all chunk IDs in the collection
        """
        try:
            # Use scroll to get all points without scoring
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust as needed
                with_payload=False,  # We only need IDs
                with_vectors=False
            )
            
            return [str(point.id) for point in points]
        except Exception as e:
            logger.error(f"Error getting all chunk IDs: {e}")
            raise