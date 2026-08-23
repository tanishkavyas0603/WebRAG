import apiClient from './client';

export const documentsApi = {
  ingest: async (url) => {
    const response = await apiClient.post('/api/documents/ingest', { url });
    return response.data;
  },
  getStatus: async (documentId) => {
    const response = await apiClient.get(`/api/documents/${documentId}/status`);
    return response.data;
  }
};
