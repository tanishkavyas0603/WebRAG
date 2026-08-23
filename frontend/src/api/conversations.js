import apiClient from './client';

export const conversationsApi = {
  create: async (documentId) => {
    const response = await apiClient.post('/api/conversations', { document_id: documentId });
    return response.data;
  },
  list: async () => {
    const response = await apiClient.get('/api/conversations');
    return response.data;
  },
  get: async (conversationId) => {
    const response = await apiClient.get(`/api/conversations/${conversationId}`);
    return response.data;
  },
  delete: async (conversationId) => {
    const response = await apiClient.delete(`/api/conversations/${conversationId}`);
    return response.data;
  },
  sendMessage: async (conversationId, message) => {
    const response = await apiClient.post(`/api/conversations/${conversationId}/messages`, { message });
    return response.data;
  },
  getMessages: async (conversationId) => {
    const response = await apiClient.get(`/api/conversations/${conversationId}/messages`);
    return response.data;
  }
};
