#!/usr/bin/env python3
"""
Database Setup Script for ECESSP Frontend

This script sets up the PostgreSQL database for the ECESSP frontend application.
It creates the necessary database, user, and tables with proper permissions.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
import sys

def setup_database():
    """Set up the ECESSP database with proper permissions."""
    
    # Database connection parameters
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'user': 'postgres',  # Using postgres superuser for setup
        'password': 'postgres',  # Default PostgreSQL password
        'database': 'postgres'  # Connect to default database first
    }
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**db_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        print("Setting up ECESSP database...")
        
        # Create database if it doesn't exist
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'ecessp_db'")
        if not cursor.fetchone():
            cursor.execute("CREATE DATABASE ecessp_db")
            print("✓ Created database 'ecessp_db'")
        else:
            print("✓ Database 'ecessp_db' already exists")
        
        # Create user if it doesn't exist
        cursor.execute("SELECT 1 FROM pg_user WHERE usename = 'ecessp_app'")
        if not cursor.fetchone():
            cursor.execute("CREATE USER ecessp_app WITH PASSWORD 'app123'")
            print("✓ Created user 'ecessp_app'")
        else:
            print("✓ User 'ecessp_app' already exists")
        
        # Grant permissions
        cursor.execute("GRANT ALL PRIVILEGES ON DATABASE ecessp_db TO ecessp_app")
        cursor.execute("GRANT ALL PRIVILEGES ON SCHEMA public TO ecessp_app")
        cursor.execute("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ecessp_app")
        cursor.execute("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ecessp_app")
        cursor.execute("ALTER USER ecessp_app CREATEDB")  # Allow user to create tables
        
        print("✓ Granted all necessary permissions to 'ecessp_app'")
        
        # Test connection with the new user
        test_params = {
            'host': 'localhost',
            'port': 5432,
            'user': 'ecessp_app',
            'password': 'app123',
            'database': 'ecessp_db'
        }
        
        test_conn = psycopg2.connect(**test_params)
        test_cursor = test_conn.cursor()
        test_cursor.execute("SELECT version()")
        version = test_cursor.fetchone()[0]
        print(f"✓ Successfully connected as 'ecessp_app': {version}")
        
        test_conn.close()
        cursor.close()
        conn.close()
        
        print("\n🎉 Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: cd ecessp-frontend && npm run db:push")
        print("2. The application should now work with PostgreSQL!")
        
    except psycopg2.Error as e:
        print(f"❌ Database setup failed: {e}")
        print("\nPossible solutions:")
        print("1. Make sure PostgreSQL is running on localhost:5432")
        print("2. Check that the postgres user password is correct")
        print("3. Ensure you have the necessary permissions")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("ECESSP Database Setup")
    print("=" * 50)
    setup_database()