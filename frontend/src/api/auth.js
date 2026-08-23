import apiClient from './client';

export const authApi = {
  register: async (email, password) => {
    const response = await apiClient.post('/api/auth/register', { email, password });
    return response.data;
  },
  login: async (username, password) => {
    const params = new URLSearchParams();
    params.append('username', username);
    params.append('password', password);
    
    const response = await apiClient.post('/api/auth/login', params, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    });
    return response.data;
  }
};
