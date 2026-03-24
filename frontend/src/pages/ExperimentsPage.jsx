import { useState, useEffect } from "react";
import {
  Box, Typography, Button, Table, TableHead, TableRow, TableCell,
  TableBody, Chip, CircularProgress, Paper
} from "@mui/material";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
import { getExperiments, trainModel } from "../services/mlopsApi";

export default function ExperimentsPage() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);

  const fetchRuns = async () => {
    setLoading(true);
    try {
      const { data } = await getExperiments();
      setRuns(data);
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    try {
      await trainModel({ C: 1.0, max_iter: 100 });
      await fetchRuns();
    } finally {
      setTraining(false);
    }
  };

  useEffect(() => { fetchRuns(); }, []);

  const chartData = [...runs].reverse().map((r, i) => ({
    run: i + 1,
    accuracy: +(r.metrics.accuracy * 100).toFixed(1),
    f1: +(r.metrics.f1_score * 100).toFixed(1),
  }));

  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
        <Typography variant="h6">Experiment Runs</Typography>
        <Button variant="contained" onClick={handleTrain} disabled={training}>
          {training ? <CircularProgress size={20} /> : "Train New Run"}
        </Button>
      </Box>

      {chartData.length > 0 && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>Accuracy & F1 Over Runs</Typography>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="run" label={{ value: "Run", position: "insideBottom", offset: -2 }} />
              <YAxis domain={[0, 100]} unit="%" />
              <Tooltip formatter={(v) => `${v}%`} />
              <Line type="monotone" dataKey="accuracy" stroke="#1976d2" name="Accuracy" dot />
              <Line type="monotone" dataKey="f1" stroke="#388e3c" name="F1" dot />
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      )}

      {loading ? (
        <CircularProgress />
      ) : (
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Run ID</TableCell>
              <TableCell>Time</TableCell>
              <TableCell>Accuracy</TableCell>
              <TableCell>F1</TableCell>
              <TableCell>Status</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {runs.map((r) => (
              <TableRow key={r.run_id}>
                <TableCell sx={{ fontFamily: "monospace", fontSize: 11 }}>{r.run_id.slice(0, 8)}…</TableCell>
                <TableCell>{new Date(r.start_time).toLocaleString()}</TableCell>
                <TableCell>{r.metrics.accuracy ? `${(r.metrics.accuracy * 100).toFixed(1)}%` : "—"}</TableCell>
                <TableCell>{r.metrics.f1_score ? `${(r.metrics.f1_score * 100).toFixed(1)}%` : "—"}</TableCell>
                <TableCell>
                  <Chip label={r.status} size="small" color={r.status === "FINISHED" ? "success" : "default"} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Box>
  );
}
