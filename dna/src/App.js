import { useState } from 'react';
import './App.css';

function App() {
  const [sequence, setSequence] = useState('');
  const [analysis, setAnalysis] = useState('');

  const analyzeSequence = (input) => {
    // Basic DNA sequence validation and analysis
    const validSequence = /^[ATCG]+$/i.test(input);
    
    if (!validSequence) {
      setAnalysis('Invalid sequence. Please enter only A, T, C, or G characters.');
      return;
    }

    const stats = {
      length: input.length,
      a: (input.match(/A/gi) || []).length,
      t: (input.match(/T/gi) || []).length,
      c: (input.match(/C/gi) || []).length,
      g: (input.match(/G/gi) || []).length
    };

    setAnalysis(`
      Sequence Length: ${stats.length}
      A count: ${stats.a}
      T count: ${stats.t}
      C count: ${stats.c}
      G count: ${stats.g}
      GC content: ${((stats.g + stats.c) / stats.length * 100).toFixed(2)}%
    `);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>DNA Sequence Analyzer</h1>
        <div className="sequence-input">
          <textarea
            placeholder="Enter your DNA sequence (A, T, C, G only)"
            value={sequence}
            onChange={(e) => {
              setSequence(e.target.value.toUpperCase());
              analyzeSequence(e.target.value.toUpperCase());
            }}
            rows={5}
            cols={50}
          />
        </div>
        {analysis && (
          <div className="analysis-results">
            <h2>Analysis Results:</h2>
            <pre>{analysis}</pre>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
