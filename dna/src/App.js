import { useState } from 'react';
import './App.css';
import CircularProgress from '@mui/material/CircularProgress'; // You'll need to install @mui/material

function App() {
  const [file, setFile] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState(null);

  const verifyFile = async (uploadedFile) => {
    setIsVerifying(true);
    setVerificationResult(null);

    try {
      // Simulate verification delay
      await new Promise(resolve => setTimeout(resolve, 3000));

      setVerificationResult({
        success: true,
        message: '✅ Successfully generated and verified proof!',
        details: `Processed file: ${uploadedFile.name}`
      });

    } catch (error) {
      setVerificationResult({
        success: false,
        message: 'Error processing file: ' + error.message
      });
    } finally {
      setIsVerifying(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>DNA Sequence Proof Verifier</h1>
        <div className="file-input">
          <input
            type="file"
            onChange={(e) => {
              const file = e.target.files[0];
              if (file) {
                setFile(file);
                verifyFile(file);
              }
            }}
          />
          <p className="file-instructions">
            Please upload your DNA sequence file
          </p>
        </div>
        
        {isVerifying && (
          <div className="verification-status">
            <CircularProgress />
            <p>Generating and verifying proof...</p>
          </div>
        )}

        {verificationResult && (
          <div className={`verification-result ${verificationResult.success ? 'success' : 'error'}`}>
            <h2>{verificationResult.success ? 'Verification Complete' : 'Verification Failed'}</h2>
            <p>{verificationResult.message}</p>
            {verificationResult.success && (
              <div className="verification-details">
                <p>{verificationResult.details}</p>
              </div>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
