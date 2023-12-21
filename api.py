from flask import Flask, request, jsonify
from main import PHI2


model_path = "../phi-2"
model_type = "CUDA_FP16"
llm = PHI2(model_path, model_type, max_length=500, temperature=0.6)

app = Flask("PHI2_API")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_input = data["user"]

    llm_response, response_time = llm.generate(user_input)

    result = {
        "message": {
            "user": user_input,
            "llm": llm_response,
        },
        "time": response_time,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run()
