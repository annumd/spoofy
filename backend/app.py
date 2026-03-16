@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["audio"]

        filename = str(uuid.uuid4()) + "_" + file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # extract features
        features = extract_features(file_path)

        if len(features) != 59:
            return jsonify({"error": "Feature extraction failed"}), 400

        features = np.array(features).reshape(1, -1)

        # prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0].max()
        confidence = round(confidence * 100, 2)

        result = "Spoofed Voice" if prediction == 1 else "Genuine Voice"

        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            "result": result,
            "confidence": f"{confidence}%"
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)})
