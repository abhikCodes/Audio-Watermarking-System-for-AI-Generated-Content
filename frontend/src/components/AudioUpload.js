import React, { useRef, useState, useEffect } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
// Import the default export from wavesurfer.js
import wavesurfer from 'wavesurfer.js';

const AudioUpload = ({ onAudioUploaded, isLoading = false, label = "Upload Audio File", accept = "audio/wav,audio/mp3,audio/mpeg" }) => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState("");
  const waveformRef = useRef(null);
  const wavesurferRef = useRef(null);
  const audioRef = useRef(null);
  const audioUrlRef = useRef(null);

  const onDrop = (acceptedFiles) => {
    const selectedFile = acceptedFiles[0];
    if (!selectedFile.type.startsWith('audio/')) {
      setError('Please upload an audio file.');
      return;
    }
    setError('');
    setFile(selectedFile);
    const audioUrl = URL.createObjectURL(selectedFile);
    audioUrlRef.current = audioUrl;
    if (waveformRef.current) {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
      }
      const wavesurferInstance = wavesurfer.create({
        container: waveformRef.current,
        waveColor: '#3f51b5',
        progressColor: '#f50057',
        cursorColor: '#333',
        barWidth: 2,
        barRadius: 3,
        height: 60,
        normalize: true,
      });
      wavesurferInstance.load(audioUrl);
      wavesurferRef.current = wavesurferInstance;
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
      }
    }
    onAudioUploaded(selectedFile);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: accept,
    multiple: false
  });

  useEffect(() => {
    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
      }
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
    };
  }, []);

  return (
    <Box sx={{ width: '100%' }}>
      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.400',
          borderRadius: 2,
          p: 2,
          textAlign: 'center',
          backgroundColor: isDragActive ? 'rgba(63,81,181,0.05)' : 'background.paper',
          cursor: 'pointer',
          transition: 'all 0.2s',
          mb: 1,
        }}
      >
        <input {...getInputProps()} />
        <CloudUploadIcon sx={{ fontSize: 36, color: isDragActive ? 'primary.main' : 'grey.500', mb: 1 }} />
        <Typography variant="body1" color={isDragActive ? 'primary' : 'text.secondary'}>
          {isDragActive ? 'Drop the audio file here...' : label}
        </Typography>
        {error && (
          <Typography variant="body2" color="error" sx={{ mt: 1 }}>
            {error}
          </Typography>
        )}
        {isLoading && <CircularProgress size={22} sx={{ mt: 2 }} />}
      </Box>
      {file && (
        <Box sx={{ mt: 1 }}>
          <Typography variant="subtitle2" sx={{ mb: 1, textAlign: 'center' }}>
            {file.name}
          </Typography>
          <Box ref={waveformRef} sx={{ width: '100%', minHeight: 40, mb: 1 }} />
          <audio ref={audioRef} style={{ display: 'none' }} />
        </Box>
      )}
    </Box>
  );
};

export default AudioUpload;