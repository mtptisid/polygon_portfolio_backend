#!/usr/bin/env python3
"""
Quick test to verify the app can be imported without errors.
Run this before deploying to catch import issues early.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_import():
    """Test if main app can be imported"""
    try:
        logger.info("Testing app import...")
        from main import app
        logger.info("✓ App imported successfully!")
        logger.info(f"✓ App title: {app.title}")
        logger.info(f"✓ App version: {app.version}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import app: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_import()
    sys.exit(0 if success else 1)
