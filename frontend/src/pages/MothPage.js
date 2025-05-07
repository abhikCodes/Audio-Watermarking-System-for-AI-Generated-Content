import React, { useState} from 'react';
import { Box, Card, CardContent, Button, Alert, CircularProgress, Typography, Grid, Paper, useTheme, alpha } from '@mui/material';
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
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';

  const handleAudioUploaded = (file) => {
    setAudioFile(file);
    setWatermarkedAudio(null);
    setError('');
    setSuccess('');
    setLoading(false);
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
      formData.append('file', audioFile);

      // Call the watermark endpoint
      const response = await axios.post(`${API_BASE_URL}/watermark`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob'
      });
      
      // Create a URL for the downloaded blob
      const audioUrl = URL.createObjectURL(response.data);
      setWatermarkedAudio(audioUrl);
      setSuccess('Audio successfully watermarked!');
      setLoading(false);
    } catch (err) {
      console.error('Watermarking error:', err);
      setError('Failed to watermark audio. ' + (err.response?.data?.detail || err.message));
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
          backgroundColor: theme => alpha(theme.palette.background.paper, 0.7),
          backdropFilter: 'blur(10px)',
          boxShadow: '0 10px 30px rgba(0, 0, 0, 0.08)',
          overflow: 'hidden'
        }}
      >
        <Box 
          sx={{
            position: 'absolute',
            top: -20,
            right: -20,
            width: 200,
            height: 200,
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(0,0,0,0.03) 0%, rgba(0,0,0,0) 70%)',
            zIndex: 0
          }}
        />
        
        <Box sx={{ position: 'relative', zIndex: 1 }}>
          <Typography variant="h5" sx={{ mb: 4, fontWeight: 600, textAlign: 'center' }}>
            Moth - Watermark Your Audio
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
            <Box 
              component="img" 
              src="/images/moth.png" 
              alt="Moth"
              className="transparent-image" 
              sx={{ 
                width: 180, 
                height: 'auto',
                filter: isDarkMode ? 'invert(1)' : 'none',
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'scale(1.05)',
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
              sx={{ 
                mt: 3, 
                mb: 2,
                py: 1.5,
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                '&:hover': {
                  boxShadow: '0 6px 16px rgba(0,0,0,0.2)',
                  transform: 'translateY(-2px)'
                },
                transition: 'all 0.3s ease'
              }}
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
                    <Card variant="outlined" sx={{ 
                      height: '100%', 
                      borderRadius: '16px',
                      boxShadow: '0 4px 16px rgba(0,0,0,0.05)',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        boxShadow: '0 6px 20px rgba(0,0,0,0.08)',
                        transform: 'translateY(-2px)'
                      }
                    }}>
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
                    <Card variant="outlined" sx={{ 
                      height: '100%',
                      borderRadius: '16px',
                      boxShadow: '0 4px 16px rgba(0,0,0,0.05)',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        boxShadow: '0 6px 20px rgba(0,0,0,0.08)',
                        transform: 'translateY(-2px)'
                      }
                    }}>
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom fontWeight={500}>
                          Watermarked Audio
                        </Typography>
                        <Box sx={{ mb: 2 }}>
                          <audio controls style={{ width: '100%' }} src={watermarkedAudio} />
                        </Box>
                        <Button
                          variant="outlined"
                          color="primary"
                          fullWidth
                          size="medium"
                          startIcon={<DownloadIcon />}
                          onClick={() => {
                            const link = document.createElement('a');
                            link.href = watermarkedAudio;
                            link.download = 'watermarked_audio.wav';
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                          }}
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
        </Box>
      </Paper>
    </Box>
  );
};

export default MothPage;