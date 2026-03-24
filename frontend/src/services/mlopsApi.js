import axios from "axios";

const api = axios.create({ baseURL: process.env.REACT_APP_API_URL });

export const trainModel = (params) => api.post("/train", { params });
export const predictText = (text) => api.post("/predict", { text });
export const getExperiments = () => api.get("/experiments");
export const getVersions = () => api.get("/versions");
export const getMetrics = () => api.get("/metrics");
export const getDrift = () => api.get("/drift");
