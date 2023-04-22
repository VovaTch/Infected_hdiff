import {useState, useEffect} from 'react';
import Player from './components/Player'

function App() {
//Main app function

  const [soundClip, setSoundClip] = useState({
    waveform_img_path: require("./output/img/sample_img.png"),
    src_path: require("./output/sound/sample.mp3"),
  })

  return (
    <div className="App">
      <h1>Infected HDiff</h1>
      <h2>Hierarchical Music Generation Model</h2>
      <Player 
        waveform_img={soundClip.waveform_img_path} 
        waveform_sound={soundClip.src_path}
      />
    </div>
  );
}

export default App;
