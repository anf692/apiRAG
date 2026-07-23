from rest_framework import serializers

class RAGRequestSerializer(serializers.Serializer):
    question = serializers.CharField(
        required=True,
        help_text="Pose ta question sur le règlement"
    )
