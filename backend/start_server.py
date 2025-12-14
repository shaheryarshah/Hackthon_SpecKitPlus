import uvicorn
import sys
import os

# Add the project root to the path so modules can be imported properly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def main():
    # Import inside the function after modifying sys.path
    from backend.src.db.database import init_db
    import backend.src.main as main_module

    # Initialize the database
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")

    # Start the server
    print("Starting the server...")
    uvicorn.run(
        main_module.app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()