from drf_spectacular.utils import extend_schema
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response

from .serializers import RAGRequestSerializer
from .services.rag_pipeline import run_rag, evaluate, multilingual_rag


class RAGAPIView(GenericAPIView):
    serializer_class = RAGRequestSerializer

    @extend_schema(request=RAGRequestSerializer)
    def post(self, request):
        serializer = self.get_serializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        question = serializer.validated_data["question"]

        result = multilingual_rag(question)

        evaluation = evaluate(
            result["question_fr"],
            result["context"],
            result["reponse_fr"]
        )

        return Response({
            "question": result["question_originale"],
            "question_fr": result["question_fr"],
            "answer_fr": result["reponse_fr"],
            "answer_wolof": result["reponse_wolof"],
            "evaluation": evaluation
        })

    
    
