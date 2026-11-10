"""
WSGI config for cancer_benchmark project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cancer_benchmark.settings')

application = get_wsgi_application()
