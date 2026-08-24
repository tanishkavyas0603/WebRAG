import apiClient from './client';

export const documentsApi = {
  ingest: async (url) => {
    const response = await apiClient.post('/api/documents/ingest', { url });
    console.log("[INGEST] POST response:", response.data);
    console.log("[INGEST] document ID:", response.data.id);
    return response.data;
  },
  getStatus: async (documentId) => {
    console.log("[INGEST] polling document:", documentId);
    const response = await apiClient.get(`/api/documents/${documentId}/status`);
    return response.data;
  }
};
