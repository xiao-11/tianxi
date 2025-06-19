import io
import re
import base64

from openai import OpenAI
from PIL import Image

from model_adapters import BaseAdapter
import pdb


class UITarsAdapter(BaseAdapter):
    def __init__(
        self, 
        client: OpenAI,
        model: str,
    ):
        self.client = client
        self.model = model

    def generate(
        self,
        query: str,
        image: str
    ) -> str:
        with open(image, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        # image_data = io.BytesIO()
        # image.save(image_data, format="PNG")
        # image = base64.b64encode(image_data.getvalue()).decode('utf-8')

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpg;base64,{image_data}"}
                            },
                        ],
                        "response_formate": {"type": "json_object"}
                    }
                ],
                max_tokens=512,
                temperature=0.0
            )
            outputs = response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            outputs = ""

        return outputs
