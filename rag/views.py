from drf_spectacular.utils import extend_schema
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response

from .serializers import RAGRequestSerializer
from .services.rag_pipeline import run_rag, evaluate


class RAGAPIView(GenericAPIView):
    serializer_class = RAGRequestSerializer

    @extend_schema(request=RAGRequestSerializer)
    def post(self, request):
        serializer = self.get_serializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        question = serializer.validated_data["question"]

        answer, context = run_rag(question)
        evaluation = evaluate(question, context, answer)

        return Response({
            "question": question,
            "answer": answer,
            "evaluation": evaluation
        })

