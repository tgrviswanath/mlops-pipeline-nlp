import { useState } from "react";
import {
  Box, Typography, TextField, Button, Paper, Chip, CircularProgress, LinearProgress
} from "@mui/material";
import { predictText } from "../services/mlopsApi";

const EXAMPLES = [
  "This product is absolutely amazing!",
  "Terrible quality, very disappointed.",
  "It's okay, nothing special.",
  "Best purchase I've ever made.",
];

export default function PredictPage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePredict = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    try {
      const { data } = await predictText(text);
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || "Prediction failed — train a model first.");
    } finally {
      setLoading(false);
    }
  };

  const colorMap = { positive: "success", negative: "error", neutral: "default" };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>Live Inference</Typography>

      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 2 }}>
        {EXAMPLES.map((ex) => (
          <Chip key={ex} label={ex.slice(0, 30) + "…"} onClick={() => setText(ex)} variant="outlined" size="small" />
        ))}
      </Box>

      <TextField
        fullWidth multiline rows={3}
        label="Enter text to classify"
        value={text}
        onChange={(e) => setText(e.target.value)}
        sx={{ mb: 2 }}
      />

      <Button variant="contained" onClick={handlePredict} disabled={loading || !text.trim()}>
        {loading ? <CircularProgress size={20} /> : "Predict"}
      </Button>

      {error && <Typography color="error" sx={{ mt: 2 }}>{error}</Typography>}

      {result && (
        <Paper sx={{ p: 3, mt: 3 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
            <Typography variant="subtitle1">Prediction:</Typography>
            <Chip label={result.prediction} color={colorMap[result.prediction] || "default"} />
          </Box>
          <Typography variant="body2" gutterBottom>
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </Typography>
          <LinearProgress
            variant="determinate"
            value={result.confidence * 100}
            color={colorMap[result.prediction] || "primary"}
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Paper>
      )}
    </Box>
  );
}
