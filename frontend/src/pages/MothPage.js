import React, { useState } from 'react';
import { Box, Card, CardContent, Button, Alert, CircularProgress, Typography, Grid } from '@mui/material';
import AudioUpload from '../components/AudioUpload';
import axios from 'axios';
import DownloadIcon from '@mui/icons-material/Download';
import EnhancedEncryptionIcon from '@mui/icons-material/EnhancedEncryption';

// API base URL - update this to match your backend URL
const API_BASE_URL = 'http://localhost:8000';

const MothPage = () => {
  const [audioFile, setAudioFile] = useState(null);
  const [watermarkedAudio, setWatermarkedAudio] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleAudioUploaded = (file) => {
    setAudioFile(file);
    setWatermarkedAudio(null);
    setError('');
    setSuccess('');
  };

  const handleWatermark = async () => {
    if (!audioFile) {
      setError('Please upload an audio file first.');
      return;
    }
    setLoading(true);
    setError('');
    setSuccess('');
    try {
      const formData = new FormData();
      formData.append('file', audioFile); // 'file' matches the parameter name in FastAPI

      // Call the actual backend API
      const response = await axios.post(`${API_BASE_URL}/process-audio/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      // Get result data from the response
      const resultData = response.data;
      
      // Get the download URL for the watermarked file
      const watermarkedUrl = `${API_BASE_URL}/download-processed/${resultData.file_id}/${resultData.original_filename}`;
      
      // Fetch the audio file as blob
      const audioResponse = await fetch(watermarkedUrl);
      const audioBlob = await audioResponse.blob();
      
      // Create a URL for the downloaded blob
      const audioUrl = URL.createObjectURL(audioBlob);
      setWatermarkedAudio(audioUrl);
      setSuccess('Audio successfully watermarked!');
      setLoading(false);
    } catch (err) {
      console.error('Watermarking error:', err);
      setError('Failed to watermark audio. ' + (err.response?.data?.detail || err.message));
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (watermarkedAudio) {
      const link = document.createElement('a');
      link.href = watermarkedAudio;
      link.download = 'watermarked_audio.wav';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', mt: 6 }}>
      <Card elevation={3}>
        <CardContent>
          <Typography variant="h5" sx={{ mb: 4, fontWeight: 600, textAlign: 'center' }}>
            Moth - Watermark Your Audio
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
            <Box 
              component="img" 
              src="/moth.png" 
              alt="Moth" 
              sx={{ 
                width: 200, 
                height: 'auto',
                filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.2))',
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'scale(1.05)',
                  filter: 'drop-shadow(0 6px 12px rgba(0,0,0,0.3))',
                }
              }} 
            />
          </Box>
          
          <AudioUpload 
            onAudioUploaded={handleAudioUploaded} 
            isLoading={loading} 
            label="Drag & Drop Your Audio File Here (WAV)" 
          />
          
          {audioFile && (
            <Button
              variant="contained"
              color="primary"
              fullWidth
              size="large"
              onClick={handleWatermark}
              disabled={loading || !audioFile}
              startIcon={loading ? null : <EnhancedEncryptionIcon />}
              sx={{ mt: 3, mb: 2 }}
            >
              {loading ? <CircularProgress size={22} color="inherit" /> : 'Apply AI Watermark'}
            </Button>
          )}
          
          {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
          {success && <Alert severity="success" sx={{ mt: 2 }}>{success}</Alert>}
          
          {(audioFile || watermarkedAudio) && (
            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 500 }}>
                Compare Audio Files
              </Typography>
              
              <Grid container spacing={2}>
                {audioFile && (
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined" sx={{ height: '100%' }}>
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom fontWeight={500}>
                          Original Audio
                        </Typography>
                        <Box sx={{ mb: 2 }}>
                          <audio controls style={{ width: '100%' }} src={URL.createObjectURL(audioFile)} />
                        </Box>
                        <Button
                          variant="outlined"
                          color="primary"
                          fullWidth
                          size="medium"
                          startIcon={<DownloadIcon />}
                          onClick={() => {
                            const link = document.createElement('a');
                            link.href = URL.createObjectURL(audioFile);
                            link.download = 'original_audio.wav';
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                          }}
                        >
                          Download Original
                        </Button>
                      </CardContent>
                    </Card>
                  </Grid>
                )}
                
                {watermarkedAudio && (
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined" sx={{ height: '100%' }}>
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom fontWeight={500}>
                          Watermarked Audio
                        </Typography>
                        <Box sx={{ mb: 2 }}>
                          <audio controls style={{ width: '100%' }} src={watermarkedAudio} />
                        </Box>
                        <Button
                          variant="contained"
                          color="secondary"
                          fullWidth
                          size="medium"
                          startIcon={<DownloadIcon />}
                          onClick={handleDownload}
                        >
                          Download Watermarked
                        </Button>
                      </CardContent>
                    </Card>
                  </Grid>
                )}
              </Grid>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default MothPage;