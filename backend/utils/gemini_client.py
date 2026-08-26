import os
from google import genai


class GeminiClient:
    """Client for Google Gemini API integration"""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")

        if self.api_key:
            print("✅ Gemini API key loaded")
        else:
            print("⚠️ No Gemini API key found in environment")

        self.client = (
            genai.Client(api_key=self.api_key)
            if self.api_key
            else None
        )

        self.model = "gemini-2.5-flash"

    async def get_response(
        self,
        message: str,
        context: str = "",
        language: str = "en"
    ) -> str:

        if not self.client:
            return self._get_fallback_response(
                message,
                context,
                language
            )

        try:
            system_prompt = self._build_system_prompt(language)

            user_message = (
                f"{context}\n\n{message}"
                if context
                else message
            )

            full_prompt = f"""
{system_prompt}

User's question:
{user_message}
"""

            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt
            )

            if response.text:
                return response.text

            return self._get_fallback_response(
                message,
                context,
                language
            )

        except Exception as e:
            print(
                f"❌ Gemini API exception: "
                f"{type(e).__name__}: {e}"
            )

            return self._get_fallback_response(
                message,
                context,
                language
            )

    def _build_system_prompt(self, language: str) -> str:

        if language == "hi":
            return """
आप एक कृषि विशेषज्ञ हैं जो किसानों को पौधों की बीमारियों,
उपचार और कृषि पद्धतियों के बारे में सलाह देते हैं।

सरल हिंदी में जवाब दें जिसे किसान आसानी से समझ सकें।

व्यावहारिक और कार्रवाई योग्य सलाह दें।
"""

        return """
You are an agricultural expert helping farmers with
plant diseases, treatments, and farming practices.

Provide clear, practical, and actionable advice.

Keep responses concise and easy to understand.

When discussing plant diseases, explain:
1. What the disease is
2. Common symptoms
3. Possible causes
4. Treatment
5. Prevention
"""

    def _get_fallback_response(
        self,
        message: str,
        context: str,
        language: str
    ) -> str:

        if language == "hi":
            return """
मैं अभी Gemini AI से जुड़ नहीं पा रहा हूं।

कृपया कुछ समय बाद पुनः प्रयास करें।

सामान्य सुझाव:
- संक्रमित पत्तियों को हटा दें
- पौधों के बीच उचित दूरी बनाए रखें
- अत्यधिक पानी देने से बचें
- उचित अनुशंसित उपचार का उपयोग करें
"""

        return """
I'm currently unable to connect to Google Gemini.

Please try again later.

General recommendations:
- Remove infected leaves
- Maintain proper plant spacing
- Avoid overwatering
- Use appropriate recommended treatment
"""