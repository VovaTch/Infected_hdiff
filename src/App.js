import { useState } from 'react';
import Player from './components/Player'
import ProgressBar from './components/GenerationProgressBars';

function App() {
  //Main app function

  // Set states
  const [soundClip, setSoundClip] = useState({
    waveform_img_path: require("./output/img/sample_img.png"),
    src_path: require("./output/sound/sample.mp3"),
  })
  const [progressBarCol, setProgressBarCol] = useState([]);

  // Helper functions
  function addBar(name, color) {

    const passingProps = {
      barSettings: { name: name, color: color },
      barCompleted: 55,
      id: Math.floor(Math.random() * 100000),
    };
    const newProgressBar = <ProgressBar
      barSettings={passingProps.barSettings}
      barCompleted={passingProps.barCompleted}
      id={passingProps.id}
    />;
    setProgressBarCol(oldList => [...oldList, newProgressBar])

  };

  // // TODO: Testing

  return (
    <div className="App">
      <h1>Infected HDiff</h1>
      <h2>Hierarchical Music Generation Model</h2>
      <Player
        waveform_img={soundClip.waveform_img_path}
        waveform_sound={soundClip.src_path}
        progress_bar_col={progressBarCol}
        addBar={addBar}
      />
      <ul>
        {progressBarCol.map((pBar) => {
          return (<li>{pBar}</li>)
        })}
      </ul>
    </div>
  );
}

export default App;
