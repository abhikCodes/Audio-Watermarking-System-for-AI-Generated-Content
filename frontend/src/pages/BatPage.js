import React, { useState } from 'react';
import { Box, Card, CardContent, Button, Alert, CircularProgress, Typography, Grid } from '@mui/material';
import AudioUpload from '../components/AudioUpload';
import axios from 'axios';
import DownloadIcon from '@mui/icons-material/Download';
import SearchIcon from '@mui/icons-material/Search';

// API base URL - update this to match your backend URL
const API_BASE_URL = 'http://localhost:8000';

const BatPage = () => {
  const [audioFile, setAudioFile] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleAudioUploaded = (file) => {
    setAudioFile(file);
    setDetectionResult(null);
    setError('');
    setSuccess('');
  };

  const handleDetect = async () => {
    if (!audioFile) {
      setError('Please upload an audio file first.');
      return;
    }
    setLoading(true);
    setError('');
    setSuccess('');
    setDetectionResult(null);
    try {
      const formData = new FormData();
      formData.append('file', audioFile); // 'file' matches the parameter name in FastAPI
      
      // Call the actual backend API
      const response = await axios.post(`${API_BASE_URL}/detect-watermark/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      // Process the response
      const resultData = response.data;
      setDetectionResult({
        isAIGenerated: resultData.is_watermarked,
        confidence: resultData.watermark_probability,
        source: resultData.is_watermarked ? "Moth AI Watermark" : "Natural or Unwatermarked Audio"
      });
      
      setSuccess('Audio analysis complete!');
      setLoading(false);
    } catch (err) {
      console.error('Detection error:', err);
      setError('Failed to analyze audio. ' + (err.response?.data?.detail || err.message));
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', mt: 6 }}>
      <Card elevation={3}>
        <CardContent>
          <Typography variant="h5" sx={{ mb: 4, fontWeight: 600, textAlign: 'center' }}>
            Bat - Detect AI-Generated Audio
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
            <Box 
              component="img" 
              src="/bat.png" 
              alt="Bat" 
              sx={{ 
                width: 220, 
                height: 'auto',
                filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.2))',
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'scale(1.05) rotate(3deg)',
                  filter: 'drop-shadow(0 6px 12px rgba(0,0,0,0.3))',
                }
              }} 
            />
          </Box>
          
          <AudioUpload 
            onAudioUploaded={handleAudioUploaded} 
            isLoading={loading} 
            label="Drag & Drop Audio to Analyze" 
          />
          
          {audioFile && (
            <Button
              variant="contained"
              color="primary"
              fullWidth
              size="large"
              onClick={handleDetect}
              disabled={loading || !audioFile}
              startIcon={loading ? null : <SearchIcon />}
              sx={{ mt: 2 }}
            >
              {loading ? <CircularProgress size={22} color="inherit" /> : 'Analyze Audio'}
            </Button>
          )}
          
          {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
          {success && <Alert severity="success" sx={{ mt: 2 }}>{success}</Alert>}
          
          {detectionResult && (
            <Box sx={{ mt: 4, p: 3, bgcolor: 'background.default', borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, textAlign: 'center' }}>
                Detection Results
              </Typography>
              
              <Box sx={{ 
                p: 3, 
                bgcolor: detectionResult.isAIGenerated ? 'rgba(244, 67, 54, 0.08)' : 'rgba(76, 175, 80, 0.08)', 
                borderRadius: 2,
                border: '1px solid',
                borderColor: detectionResult.isAIGenerated ? 'rgba(244, 67, 54, 0.5)' : 'rgba(76, 175, 80, 0.5)',
                textAlign: 'center',
                mb: 3
              }}>
                <Typography variant="h5" sx={{ fontWeight: 600, color: detectionResult.isAIGenerated ? 'error.main' : 'success.main' }}>
                  {detectionResult.isAIGenerated ? 'AI-Generated Audio Detected!' : 'Natural Audio Detected'}
                </Typography>
                <Typography variant="body1" sx={{ mt: 1 }}>
                  Confidence: <strong>{(detectionResult.confidence * 100).toFixed(1)}%</strong>
                </Typography>
                <Typography variant="body1" sx={{ mt: 1 }}>
                  Source: <strong>{detectionResult.source}</strong>
                </Typography>
              </Box>
              
              <Box sx={{ mt: 4 }}>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 500 }}>
                  Uploaded Audio
                </Typography>
                
                <Card variant="outlined">
                  <CardContent>
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
                        link.download = 'analyzed_audio.wav';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                      }}
                    >
                      Download Audio
                    </Button>
                  </CardContent>
                </Card>
              </Box>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default BatPage;