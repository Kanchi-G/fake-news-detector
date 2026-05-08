async function checkNews() {
    const text = document.getElementById("newsText").value;

    if (!text) {
        document.getElementById("result").innerText = "❌ Please enter text";
        return;
    }

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (data.error) {
            document.getElementById("result").innerText = "❌ " + data.error;
            document.getElementById("confidence").innerText = "";
        } else {
            document.getElementById("result").innerText = data.prediction;
            document.getElementById("confidence").innerText = "Confidence: " + data.confidence;
        }

    } catch (error) {
        console.error(error);
        document.getElementById("result").innerText = "❌ Server error";
    }
}
