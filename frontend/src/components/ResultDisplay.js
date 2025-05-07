import React from 'react';
import { Box, Paper, Typography, LinearProgress, Button } from '@mui/material';
import { motion } from 'framer-motion';
import { alpha } from '@mui/material';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';

const ResultDisplay = ({ 
  title, 
  result, 
  downloadUrl, 
  downloadFilename, 
  isPending = false,
  resultColor = '#673ab7',
  icon: Icon,
  description
}) => {
  return (
    <Paper
      component={motion.div}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      elevation={3}
      sx={{ 
        p: 3, 
        mt: 3, 
        borderRadius: 3,
        position: 'relative',
        overflow: 'hidden',
        // bgcolor: '#fff',
        bgcolor: 'background.paper',
      }}
    >
      {/* Background decorative element */}
      <Box 
        sx={{ 
          position: 'absolute', 
          top: -100, 
          right: -100, 
          width: 200, 
          height: 200, 
          borderRadius: '50%', 
          // background: `radial-gradient(circle, ${resultColor}20 0%, transparent 70%)`,
          background: theme => `radial-gradient(circle, ${alpha(resultColor, 0.2)} 0%, transparent 70%)`,
          zIndex: 0
        }} 
      />
      
      <Box sx={{ position: 'relative', zIndex: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          {Icon && <Icon sx={{ fontSize: 32, color: resultColor, mr: 1 }} />}
          <Typography variant="h5" component="h2" color="primary" gutterBottom>
            {title}
          </Typography>
        </Box>
        
        {isPending ? (
          <Box sx={{ width: '100%', mt: 2, mb: 2 }}>
            <LinearProgress color="secondary" />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Processing your audio...
            </Typography>
          </Box>
        ) : (
          <>
            {result && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="body1" sx={{ mb: 1 }}>
                  {description}
                </Typography>
                {typeof result === 'number' && (
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <LinearProgress 
                      variant="determinate" 
                      value={result * 100} 
                      sx={{ 
                        height: 15, 
                        borderRadius: 2, 
                        flexGrow: 1,
                        background: '#e0e0e0',
                        '& .MuiLinearProgress-bar': {
                          background: `linear-gradient(90deg, #e0e0e0 0%, ${resultColor} 100%)`,
                          borderRadius: 2,
                        }
                      }} 
                    />
                    <Typography variant="h6" sx={{ ml: 2, color: resultColor }}>
                      {(result * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                )}
                {typeof result === 'string' && (
                  <Typography variant="body1">
                    {result}
                  </Typography>
                )}
              </Box>
            )}
            {downloadUrl && (
              <Button 
                variant="contained" 
                color="secondary" 
                startIcon={<CloudDownloadIcon />}
                href={downloadUrl}
                download={downloadFilename}
                sx={{ mt: 2 }}
              >
                Download Processed Audio
              </Button>
            )}
          </>
        )}
      </Box>
    </Paper>
  );
};

export default ResultDisplay;