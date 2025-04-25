import React, { useState } from 'react';
import { Box, Card, CardContent, Button, Alert, CircularProgress, Typography, Grid, Paper, useTheme } from '@mui/material';
import AudioUpload from '../components/AudioUpload';
import axios from 'axios';
import DownloadIcon from '@mui/icons-material/Download';
import SearchIcon from '@mui/icons-material/Search';

// API base URL - update this to match your backend URL
const API_BASE_URL = 'http://localhost:8001';

const BatPage = () => {
  const [audioFile, setAudioFile] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';

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
      <Paper 
        elevation={0} 
        sx={{ 
          position: 'relative',
          mb: 4,
          borderRadius: '20px',
          p: 3,
          backgroundColor: 'rgba(255, 255, 255, 0.7)',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 10px 30px rgba(0, 0, 0, 0.08)',
          overflow: 'hidden'
        }}
      >
        <Box 
          sx={{
            position: 'absolute',
            top: -20,
            left: -20,
            width: 200,
            height: 200,
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(0,0,0,0.03) 0%, rgba(0,0,0,0) 70%)',
            zIndex: 0
          }}
        />
        
        <Box sx={{ position: 'relative', zIndex: 1 }}>
          <Typography variant="h5" sx={{ mb: 4, fontWeight: 600, textAlign: 'center' }}>
            Bat - Detect AI-Generated Audio
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
            <Box 
              component="img" 
              src="/images/bat.png" 
              alt="Bat"
              className="transparent-image" 
              sx={{ 
                width: 200, 
                height: 'auto',
                filter: isDarkMode ? 'invert(1)' : 'none',
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'scale(1.05) rotate(3deg)',
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
              sx={{ 
                mt: 3,
                py: 1.5,
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                '&:hover': {
                  boxShadow: '0 6px 16px rgba(0,0,0,0.2)',
                  transform: 'translateY(-2px)'
                },
                transition: 'all 0.3s ease'
              }}
            >
              {loading ? <CircularProgress size={22} color="inherit" /> : 'Analyze Audio'}
            </Button>
          )}
          
          {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
          {success && <Alert severity="success" sx={{ mt: 2 }}>{success}</Alert>}
          
          {detectionResult && (
            <Box sx={{ mt: 4, p: 3, bgcolor: 'rgba(255, 255, 255, 0.5)', borderRadius: 2, border: '1px solid', borderColor: 'divider', backdropFilter: 'blur(5px)' }}>
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
                mb: 3,
                boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
              }}>
                <Typography variant="h5" sx={{ fontWeight: 600, color: detectionResult.isAIGenerated ? 'error.main' : 'success.main' }}>
                  {detectionResult.isAIGenerated ? 'AI-Generated Audio Detected!' : 'Natural Audio Detected'}
                </Typography>
                <Typography variant="body1" sx={{ mt: 1 }}>
                </Typography>
                <Typography variant="body1" sx={{ mt: 1 }}>
                  Source: <strong>{detectionResult.source}</strong>
                </Typography>
              </Box>
              
              <Box sx={{ mt: 4 }}>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 500 }}>
                  Uploaded Audio
                </Typography>
                
                <Card variant="outlined" sx={{ 
                  borderRadius: '16px',
                  boxShadow: '0 4px 16px rgba(0,0,0,0.05)',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    boxShadow: '0 6px 20px rgba(0,0,0,0.08)',
                    transform: 'translateY(-2px)'
                  }
                }}>
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
        </Box>
      </Paper>
    </Box>
  );
};

export default BatPage;