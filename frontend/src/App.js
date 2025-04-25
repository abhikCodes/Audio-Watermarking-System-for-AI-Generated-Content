import React, { useState } from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Container, 
  Box, 
  Button, 
  IconButton, 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText,
  useMediaQuery,
  Divider
} from '@mui/material';
import { motion } from 'framer-motion';

// Import icons
import MenuIcon from '@mui/icons-material/Menu';
import HomeIcon from '@mui/icons-material/Home';
import GitHubIcon from '@mui/icons-material/GitHub';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';

// Import pages
import MothPage from './pages/MothPage';
import BatPage from './pages/BatPage';

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [mode, setMode] = useState('light');
  const theme = React.useMemo(() => createTheme({
    palette: {
      mode,
      primary: {
        main: '#000000',
        light: '#333333',
        dark: '#000000',
      },
      secondary: {
        main: '#222222',
        light: '#444444',
        dark: '#111111',
      },
      background: {
        default: mode === 'light' ? '#f9f9f9' : '#121212',
        paper: mode === 'light' ? '#ffffff' : '#1e1e1e',
      },
      text: {
        primary: mode === 'light' ? '#333333' : '#f3f3f3',
        secondary: mode === 'light' ? '#666666' : '#b0b0b0',
      },
    },
    typography: {
      fontFamily: '"Poppins", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: { fontWeight: 700, letterSpacing: '-0.01em' },
      h2: { fontWeight: 600, letterSpacing: '-0.01em' },
      h4: { fontWeight: 600, letterSpacing: '-0.01em' },
      h6: { fontWeight: 500 },
      button: { fontWeight: 600 },
    },
    components: {
      MuiAppBar: {
        styleOverrides: {
          root: {
            boxShadow: '0 2px 10px rgba(0, 0, 0, 0.05)',
            backdropFilter: 'blur(8px)',
            backgroundColor: mode === 'light' ? 'rgba(255,255,255,0.8)' : 'rgba(24,26,32,0.9)',
            color: mode === 'light' ? '#333333' : '#f3f3f3',
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            textTransform: 'none',
            padding: '10px 20px',
            fontWeight: 600,
            transition: 'all 0.2s ease-in-out',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
            },
          },
          containedPrimary: {
            backgroundImage: 'linear-gradient(135deg, #000000 0%, #333333 100%)',
          },
          containedSecondary: {
            backgroundImage: 'linear-gradient(135deg, #222222 0%, #444444 100%)',
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            borderRadius: 16,
            boxShadow: '0 8px 16px rgba(0, 0, 0, 0.05)',
          },
          elevation3: {
            boxShadow: '0 10px 30px rgba(0, 0, 0, 0.05)',
          }
        },
      },
      MuiContainer: {
        styleOverrides: {
          root: {
            paddingTop: 24,
            paddingBottom: 24,
          },
        },
      },
    },
    shape: {
      borderRadius: 12,
    },
  }), [mode]);
  const location = useLocation();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const toggleDrawer = (open) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
      return;
    }
    setDrawerOpen(open);
  };

  // Custom icon components using the provided images
  const MothIcon = ({ color = 'currentColor', ...props }) => (
    <Box
      component="img"
      src="/images/moth.png"
      alt="Moth"
      className="transparent-image"
      sx={{ 
        width: 24, 
        height: 24, 
        filter: mode === 'dark' ? 'invert(1)' : 'none',
        ...props.sx 
      }}
      {...props}
    />
  );

  const BatIcon = ({ color = 'currentColor', ...props }) => (
    <Box
      component="img"
      src="/images/bat.png"
      alt="Bat"
      className="transparent-image"
      sx={{ 
        width: 24, 
        height: 24, 
        filter: mode === 'dark' ? 'invert(1)' : 'none',
        ...props.sx 
      }}
      {...props}
    />
  );

  const navItems = [
    { name: 'Moth Encoder', path: '/', icon: <MothIcon /> },
    { name: 'Bat Detector', path: '/bat-detector', icon: <BatIcon /> },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        minHeight: '100vh', 
        display: 'flex', 
        flexDirection: 'column',
        background: mode === 'light' 
          ? 'linear-gradient(135deg, #f9f9f9 0%, #f4f4f4 100%)' 
          : 'linear-gradient(135deg, #121212 0%, #1a1a1a 100%)',
        backgroundAttachment: 'fixed',
      }}>
        <AppBar position="sticky" color="default" elevation={0}>
          <Container maxWidth="lg">
            <Toolbar sx={{ px: { xs: 1, sm: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
                <Typography 
                  variant="h5" 
                  component={Link} 
                  to="/"
                  sx={{ 
                    fontWeight: 700, 
                    textDecoration: 'none',
                    color: 'primary.main',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1
                  }}
                >
                  <Box
                    component="img"
                    src="/images/batmoth.png"
                    alt="Moth and Bat"
                    className="transparent-image"
                    sx={{ 
                      height: 34,
                      width: 'auto',
                      filter: mode === 'dark' ? 'invert(1)' : 'none',
                    }}
                  />
                  Moth & Bat
                </Typography>
              </Box>
              <IconButton sx={{ ml: 1 }} onClick={() => setMode(mode === 'light' ? 'dark' : 'light')} color="inherit">
                {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
              </IconButton>
              {isMobile ? (
                <IconButton 
                  edge="end" 
                  color="primary" 
                  aria-label="menu"
                  onClick={toggleDrawer(true)}
                >
                  <MenuIcon />
                </IconButton>
              ) : (
                <Box sx={{ display: 'flex', gap: 2 }}>
                  {navItems.map((item) => (
                    <Button
                      key={item.path}
                      component={Link}
                      to={item.path}
                      color="primary"
                      startIcon={item.icon}
                      variant={isActive(item.path) ? "contained" : "text"}
                      sx={{ 
                        borderRadius: '10px',
                        px: 2,
                        py: 1,
                      }}
                    >
                      {item.name}
                    </Button>
                  ))}
                </Box>
              )}
            </Toolbar>
          </Container>
        </AppBar>
        
        <Drawer
          anchor="right"
          open={drawerOpen}
          onClose={toggleDrawer(false)}
        >
          <Box
            sx={{ width: 250 }}
            role="presentation"
            onClick={toggleDrawer(false)}
            onKeyDown={toggleDrawer(false)}
          >
            <List sx={{ pt: 2 }}>
              <ListItem>
                <Typography variant="h6" color="primary" sx={{ fontWeight: 600 }}>
                  Moth & Bat
                </Typography>
              </ListItem>
              <Divider sx={{ my: 1 }} />
              {navItems.map((item) => (
                <ListItem 
                  button 
                  key={item.path} 
                  component={Link} 
                  to={item.path}
                  sx={{ 
                    borderRadius: '10px', 
                    mx: 1, 
                    backgroundColor: isActive(item.path) ? 'rgba(0, 0, 0, 0.08)' : 'transparent',
                  }}
                >
                  <ListItemIcon sx={{ color: isActive(item.path) ? 'primary.main' : 'text.secondary' }}>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText 
                    primary={item.name} 
                    primaryTypographyProps={{ 
                      fontWeight: isActive(item.path) ? 600 : 400,
                      color: isActive(item.path) ? 'primary.main' : 'text.primary'
                    }} 
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>
        
        <Container component="main" sx={{ flexGrow: 1, py: 3, display: 'flex', flexDirection: 'column' }}>
          <Routes>
            <Route path="/" element={<MothPage />} />
            <Route path="/bat-detector" element={<BatPage />} />
          </Routes>
        </Container>
        
        <Box 
          component="footer" 
          sx={{ 
            p: 4, 
            mt: 'auto', 
            bgcolor: mode === 'light' ? 'rgba(255,255,255,0.8)' : 'rgba(24,26,32,0.7)',
            textAlign: 'center',
            borderTop: '1px solid rgba(0, 0, 0, 0.05)',
            backdropFilter: 'blur(8px)',
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 3, mb: 2 }}>
            <IconButton color="primary" aria-label="github repository" component="a" href="#" target="_blank">
              <GitHubIcon />
            </IconButton>
          </Box>
          <Typography variant="body2" color="text.secondary">
            Yash & Abhik © {new Date().getFullYear()}
          </Typography>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;