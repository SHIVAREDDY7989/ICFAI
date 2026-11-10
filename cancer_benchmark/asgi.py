"""
ASGI config for cancer_benchmark project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cancer_benchmark.settings')

application = get_asgi_application()
