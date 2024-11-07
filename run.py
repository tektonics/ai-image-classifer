from app import app
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('FLASK_ENV') == 'development'
        logging.info(f"Starting application on port {port}")
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        logging.error(f"Failed to start application: {str(e)}")
        raise
