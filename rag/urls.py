from django.urls import path
from .views import RAGAPIView

urlpatterns = [
    path("rag/", RAGAPIView.as_view(), name="rag"),
]