import { useState, useEffect } from "react";
import {
  Box, Typography, Grid, Paper, Chip, Table, TableHead,
  TableRow, TableCell, TableBody, CircularProgress, Alert
} from "@mui/material";
import { getMetrics, getDrift, getVersions } from "../services/mlopsApi";

function StatCard({ label, value, color }) {
  return (
    <Paper sx={{ p: 2, textAlign: "center" }}>
      <Typography variant="h4" color={color || "primary"}>{value}</Typography>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
    </Paper>
  );
}

export default function MonitoringPage() {
  const [metrics, setMetrics] = useState(null);
  const [drift, setDrift] = useState(null);
  const [versions, setVersions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const [m, d, v] = await Promise.all([getMetrics(), getDrift(), getVersions()]);
        setMetrics(m.data);
        setDrift(d.data);
        setVersions(v.data);
      } finally {
        setLoading(false);
      }
    };
    fetch();
    const interval = setInterval(fetch, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <CircularProgress />;

  return (
    <Box>
      <Typography variant="h6" gutterBottom>System Monitoring</Typography>

      {drift?.drift_detected && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Data drift detected (score: {drift.drift_score.toFixed(3)}) — auto-retraining scheduled.
        </Alert>
      )}

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} sm={3}>
          <StatCard label="Total Predictions" value={metrics?.total_predictions ?? 0} />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard label="Total Trainings" value={metrics?.total_trainings ?? 0} color="secondary" />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            label="Avg Confidence"
            value={metrics?.avg_confidence ? `${(metrics.avg_confidence * 100).toFixed(1)}%` : "—"}
            color="success.main"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            label="Drift Score"
            value={drift?.drift_score !== undefined ? drift.drift_score.toFixed(3) : "—"}
            color={drift?.drift_detected ? "error.main" : "success.main"}
          />
        </Grid>
      </Grid>

      <Typography variant="subtitle1" gutterBottom>Model Versions</Typography>
      {versions.length === 0 ? (
        <Typography color="text.secondary">No versions registered yet. Train a model first.</Typography>
      ) : (
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Version</TableCell>
              <TableCell>Run ID</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Stage</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {versions.map((v) => (
              <TableRow key={v.version}>
                <TableCell>v{v.version}</TableCell>
                <TableCell sx={{ fontFamily: "monospace", fontSize: 11 }}>{v.run_id.slice(0, 8)}…</TableCell>
                <TableCell>{new Date(v.creation_timestamp).toLocaleString()}</TableCell>
                <TableCell>
                  <Chip label={v.current_stage || "None"} size="small" />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Box>
  );
}
