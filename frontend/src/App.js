import React, { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [sentiment, setSentiment] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSentiment("");

    try {
      const response = await fetch("https://kind-nikoletta-wardbatsh22-82d02e36.koyeb.app/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch sentiment.");
      }

      const data = await response.json();
      setSentiment(data.sentiment);
    } catch (error) {
      console.error(error);
      setSentiment("Error contacting the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Smart Sentiment Analyzer</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          rows="5"
          placeholder="Enter a product review..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          required
        ></textarea>
        <br />
        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Sentiment"}
        </button>
      </form>
      {sentiment && (
        <h3>
          Predicted Sentiment: <span className={sentiment}>{sentiment}</span>
        </h3>
      )}
    </div>
  );
}

export default App;
