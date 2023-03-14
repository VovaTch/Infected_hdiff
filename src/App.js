import {useState} from 'react';

function App() {

  const [soundClip, setSoundClip] = useState([{
    waveform_img_path: "output/img/sample_img.jpg",
    src_path: "output/sound/sample.mp3",
  }])

  return (
    <div className="App">
      Player component placeholder
    </div>
  );
}

export default App;
