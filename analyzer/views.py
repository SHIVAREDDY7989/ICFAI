"""
Views for Cancer Benchmark Application:
- index_view: Dashboard with schema guide, CSV upload, charts, and metric summary.
- download_template_view: Downloads dynamic breast cancer clinical CSV sample.
- download_chart_view: Multi-format 300 DPI chart download (PNG, JPG, JPEG).
- chat_api_view: 100% offline rule- and context-based clinical chat query handler.
"""

import os
import json
from pathlib import Path

from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, FileResponse, Http404
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie
from django.conf import settings

from . import ml_engine
from . import chat_engine


@ensure_csrf_cookie
def index_view(request):
    """
    Renders the benchmark dashboard.
    Processes CSV file uploads and executes the automated 6-model ML benchmarking pipeline.
    """
    context = {
        'feature_columns': ml_engine.FEATURE_COLUMNS,
        'target_column': ml_engine.TARGET_COLUMN,
        'benchmark_data': request.session.get('benchmark_data', None),
        'error_message': None,
        'success_message': None,
    }

    if request.method == 'POST':
        uploaded_file = request.FILES.get('dataset_file') or request.FILES.get('csv_file')
        if not uploaded_file:
            context['error_message'] = "Please select a valid .csv, .pdf, or .xlsx file to upload."
            return render(request, 'analyzer/index.html', context)

        file_name_lower = uploaded_file.name.lower()
        allowed_extensions = ('.csv', '.pdf', '.xlsx')
        if not any(file_name_lower.endswith(ext) for ext in allowed_extensions):
            context['error_message'] = "Invalid file type. Supported formats are .csv, .pdf, and .xlsx."
            return render(request, 'analyzer/index.html', context)

        try:
            benchmark_data = ml_engine.execute_full_benchmark(uploaded_file, filename=uploaded_file.name)
            # Store in session for offline context-aware chatbox queries
            request.session['benchmark_data'] = benchmark_data
            context['benchmark_data'] = benchmark_data
            context['success_message'] = (
                f"Successfully benchmarked 6 classification algorithms on {benchmark_data['dataset_stats']['total_rows']} "
                f"clinical records ({benchmark_data['dataset_stats']['train_samples']} train, "
                f"{benchmark_data['dataset_stats']['test_samples']} test)."
            )
        except ValueError as ve:
            context['error_message'] = f"Data Validation Error: {str(ve)}"
        except Exception as e:
            context['error_message'] = f"Unexpected processing error: {str(e)}"

    return render(request, 'analyzer/index.html', context)


def download_template_view(request):
    """
    Dynamically generates and downloads a valid CSV template matching the exact schema
    using scikit-learn's breast cancer dataset (10 mean features + diagnosis).
    """
    try:
        csv_content = ml_engine.get_sample_csv()
        response = HttpResponse(csv_content, content_type='text/csv; charset=utf-8')
        response['Content-Disposition'] = 'attachment; filename="breast_cancer_clinical_sample.csv"'
        return response
    except Exception as e:
        return HttpResponse(f"Error generating sample template: {str(e)}", status=500)


def download_chart_view(request, chart_name: str, ext: str):
    """
    Serves 300 DPI high-resolution figures for download in PNG, JPG, or JPEG format.
    """
    allowed_charts = ['confusion_matrices', 'accuracy_comparison']
    allowed_exts = ['png', 'jpg', 'jpeg']

    clean_chart = chart_name.strip().lower()
    clean_ext = ext.strip().lower()

    if clean_chart not in allowed_charts or clean_ext not in allowed_exts:
        raise Http404("Requested chart or format not recognized.")

    export_dir = Path(settings.MEDIA_ROOT) / 'exports'
    file_path = export_dir / f"{clean_chart}.{clean_ext}"

    if not file_path.exists():
        # If files are not generated yet, try to regenerate if session data exists
        session_data = request.session.get('benchmark_data')
        if session_data and 'results' in session_data:
            ml_engine.generate_and_save_charts(session_data['results'])

    if not file_path.exists():
        raise Http404("Chart file has not been generated yet. Please upload a dataset first.")

    content_types = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
    }

    return FileResponse(
        open(file_path, 'rb'),
        as_attachment=True,
        filename=f"{clean_chart}_300dpi.{clean_ext}",
        content_type=content_types.get(clean_ext, 'application/octet-stream')
    )


@require_http_methods(["POST"])
def chat_api_view(request):
    """
    POST endpoint for 100% offline interactive chatbox.
    Reads current session metrics and queries the offline clinical chat engine.
    """
    try:
        if request.content_type == 'application/json':
            body = json.loads(request.body.decode('utf-8'))
            prompt = body.get('message', '').strip()
        else:
            prompt = request.POST.get('message', '').strip()

        session_data = request.session.get('benchmark_data', None)
        reply = chat_engine.respond_to_query(prompt, session_data)

        return JsonResponse({
            'status': 'success',
            'reply': reply,
        })
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body.'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
