#!/usr/bin/env python3
"""
Database initialization script for Careerly application.
Run this script to set up the MySQL database and create tables.
"""

import os
import sys
from main import app
from database import create_tables, reset_database

def init_database():
    """Initialize the database with all tables"""
    try:
        print("Initializing Careerly database...")
        with app.app_context():
            create_tables(app)
            print("✓ Database initialization completed successfully!")
            print("\nNext steps:")
            print("1. Update the database connection string in database.py if needed")
            print("2. Make sure MySQL server is running")
            print("3. Create a database named 'careerly_db' in MySQL")
            print("4. Run the Flask application: python main.py")
            
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure MySQL server is running")
        print("2. Check if the database 'careerly_db' exists")
        print("3. Verify MySQL credentials in database.py")
        print("4. Install required packages: pip install -r requirements.txt")
        return False
    return True

def reset_db():
    """Reset the database (WARNING: This will delete all data!)"""
    try:
        confirm = input("⚠️  WARNING: This will delete ALL data. Type 'RESET' to confirm: ")
        if confirm.strip() == 'RESET':
            print("Resetting database...")
            with app.app_context():
                reset_database(app)
                print("✓ Database reset completed!")
        else:
            print("Database reset cancelled.")
    except Exception as e:
        print(f"✗ Error resetting database: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        reset_db()
    else:
        init_database()