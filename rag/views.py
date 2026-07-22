from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .services.rag_pipeline import run_rag

class RAGAPIView(APIView):

    def post(self, request):
        question = request.data.get("question")

        if not question:
            return Response(
                {"error": "Question manquante"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            answer = run_rag(question)

            return Response({
                "question": question,
                "answer": answer
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    