from drf_spectacular.utils import extend_schema
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import RAGRequestSerializer
from .services.rag_pipeline import RAG, retriever, evaluate


class RAGAPIView(GenericAPIView):
    serializer_class = RAGRequestSerializer

    @extend_schema(request=RAGRequestSerializer)
    def post(self, request):
        serializer = self.get_serializer(data=request.data)

        # 🔹 Validation
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        question = serializer.validated_data["question"]

        try:
            # 🔹 1. Récupérer le contexte
            context_docs = retriever.invoke(question)
            context_list = [doc.page_content for doc in context_docs]
            context = ". ".join(context_list)

            # 🔹 2. Générer la réponse
            answer = RAG(question)

            # 🔹 3. Évaluation
            evaluation = evaluate(
                question=question,
                context=context,
                answer=answer
            )

            # 🔹 4. Détection "JE NE SAIS PAS"
            is_unknown = "JE NE SAIS PAS" in answer

            return Response({
                "question": question,
                "answer": answer,
                "evaluation": evaluation,
                "confidence": "low" if is_unknown else "high"
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    