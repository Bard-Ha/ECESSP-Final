#!/usr/bin/env python3
"""
Simple Database Table Setup for ECESSP Frontend

This script creates the necessary tables in the existing PostgreSQL database
using the credentials from the .env file.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

def setup_tables():
    """Create the materials table in the existing database."""
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("❌ DATABASE_URL not found in environment variables")
        print("Please set the DATABASE_URL in your .env file")
        sys.exit(1)
    
    try:
        # Connect to the database
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        print("Setting up ECESSP database tables...")
        
        # Create materials table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS materials (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            type VARCHAR(50) NOT NULL,
            properties JSONB NOT NULL
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        
        print("✓ Created materials table")
        
        # Insert some sample data
        sample_materials = [
            ('LCO (LiCoO2)', 'cathode', '{"average_voltage": 3.9, "capacity_grav": 155, "capacity_vol": 700, "stability_charge": 4.2, "stability_discharge": 3.0}'),
            ('LFP (LiFePO4)', 'cathode', '{"average_voltage": 3.2, "capacity_grav": 160, "capacity_vol": 580, "stability_charge": 3.6, "stability_discharge": 2.5}'),
            ('NMC 811', 'cathode', '{"average_voltage": 3.8, "capacity_grav": 200, "capacity_vol": 800, "stability_charge": 4.3, "stability_discharge": 2.8}'),
            ('Graphite', 'anode', '{"average_voltage": 0.1, "capacity_grav": 372, "capacity_vol": 700, "stability_charge": 0.2, "stability_discharge": 0.0}'),
            ('Silicon', 'anode', '{"average_voltage": 0.4, "capacity_grav": 3579, "capacity_vol": 8000, "stability_charge": 0.5, "stability_discharge": 0.1}'),
            ('Lithium Metal', 'anode', '{"average_voltage": 0.0, "capacity_grav": 3860, "capacity_vol": 2000, "stability_charge": 0.1, "stability_discharge": 0.0}'),
            ('Standard Liquid', 'electrolyte', '{"stability_charge": 4.5, "stability_discharge": 0.0}'),
            ('Solid State (Sulfide)', 'electrolyte', '{"stability_charge": 5.0, "stability_discharge": 0.0}'),
            ('Polymer', 'electrolyte', '{"stability_charge": 4.8, "stability_discharge": 0.0}')
        ]
        
        # Insert sample data
        insert_query = """
        INSERT INTO materials (name, type, properties)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        
        cursor.executemany(insert_query, sample_materials)
        conn.commit()
        
        print("✓ Inserted sample materials data")
        
        # Test the connection
        cursor.execute("SELECT COUNT(*) FROM materials")
        count = cursor.fetchone()[0]
        print(f"✓ Database contains {count} materials")
        
        cursor.close()
        conn.close()
        
        print("\n🎉 Database tables setup completed successfully!")
        print("\nNext steps:")
        print("1. Restart the frontend server")
        print("2. The application should now use PostgreSQL!")
        
    except psycopg2.Error as e:
        print(f"❌ Database setup failed: {e}")
        print("\nPossible solutions:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check that the DATABASE_URL is correct")
        print("3. Ensure the database and user exist")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("ECESSP Database Tables Setup")
    print("=" * 50)
    setup_tables()