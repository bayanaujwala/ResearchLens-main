#!/bin/bash


# Set the FLASK_APP environment variable
export FLASK_APP=src/server/server.py

# Start the Flask server
flask run --host=0.0.0.0

