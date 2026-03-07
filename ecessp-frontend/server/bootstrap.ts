import 'dotenv/config';

// This file ensures dotenv is loaded before any other modules
console.log('Bootstrap: Environment variables loaded');
console.log('DATABASE_URL:', process.env.DATABASE_URL ? 'Set' : 'Not set');
console.log('NODE_ENV:', process.env.NODE_ENV);