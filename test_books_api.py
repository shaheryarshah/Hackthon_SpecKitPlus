import requests
import json
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:8000/api/books"

def test_create_book():
    """Test creating a new book version"""
    print("Testing creating a new book...")
    
    # Sample book data
    book_data = {
        "version": "v1.0.0",
        "title": "Physical AI & Humanoid Robotics Handbook",
        "published_at": datetime.now().isoformat(),
        "is_active": True
    }
    
    response = requests.post(BASE_URL, json=book_data)
    
    if response.status_code == 200:
        print(f"✓ Successfully created book: {response.json()}")
        return response.json()
    else:
        print(f"✗ Failed to create book: {response.status_code}, {response.text}")
        return None

def test_get_books():
    """Test getting all books"""
    print("\nTesting getting all books...")
    
    response = requests.get(BASE_URL)
    
    if response.status_code == 200:
        books = response.json()
        print(f"✓ Retrieved {len(books)} books: {books}")
        return books
    else:
        print(f"✗ Failed to get books: {response.status_code}, {response.text}")
        return None

def test_get_book_by_version():
    """Test getting a specific book by version"""
    print("\nTesting getting a specific book...")
    
    response = requests.get(f"{BASE_URL}/v1.0.0")
    
    if response.status_code == 200:
        book = response.json()
        print(f"✓ Retrieved book: {book}")
        return book
    elif response.status_code == 404:
        print("! Book not found (might not have been created yet)")
        return None
    else:
        print(f"✗ Failed to get book: {response.status_code}, {response.text}")
        return None

def test_activate_book():
    """Test activating a book version"""
    print("\nTesting activating a book...")
    
    response = requests.put(f"{BASE_URL}/v1.0.0/activate")
    
    if response.status_code == 200:
        book = response.json()
        print(f"✓ Activated book: {book}")
        return book
    else:
        print(f"✗ Failed to activate book: {response.status_code}, {response.text}")
        return None

def test_get_book_chunks():
    """Test getting book chunks"""
    print("\nTesting getting book chunks...")
    
    response = requests.get(f"{BASE_URL}/v1.0.0/chunks")
    
    if response.status_code == 200:
        chunks = response.json()
        print(f"✓ Retrieved {len(chunks)} chunks: {chunks[:2]}...")  # Show first 2 for brevity
        return chunks
    elif response.status_code == 404:
        print("! Book not found (might not have chunks yet)")
        return None
    else:
        print(f"✗ Failed to get book chunks: {response.status_code}, {response.text}")
        return None

def main():
    """Run all tests"""
    print("Starting Book Management API Tests\n")
    
    # Test creating a book
    created_book = test_create_book()
    
    # Test getting a specific book
    test_get_book_by_version()
    
    # Test getting all books
    test_get_books()
    
    # Test activating a book
    test_activate_book()
    
    # Test getting book chunks
    test_get_book_chunks()
    
    print("\nTests completed!")

if __name__ == "__main__":
    main()