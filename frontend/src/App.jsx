import { useState } from "react";
import { Container, Tabs, Tab, Box, Typography } from "@mui/material";
import ExperimentsPage from "./pages/ExperimentsPage";
import MonitoringPage from "./pages/MonitoringPage";
import PredictPage from "./pages/PredictPage";

export default function App() {
  const [tab, setTab] = useState(0);

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        🚀 MLOps Pipeline
      </Typography>
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 3 }}>
        <Tab label="Experiments" />
        <Tab label="Monitoring" />
        <Tab label="Predict" />
      </Tabs>
      <Box>
        {tab === 0 && <ExperimentsPage />}
        {tab === 1 && <MonitoringPage />}
        {tab === 2 && <PredictPage />}
      </Box>
    </Container>
  );
}
