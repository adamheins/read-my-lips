# Read My Lips
Lip reading is hard.

## Directory structure
The `pre_processing.py` script is setup to handle a directory structure of:

```
data/align/<speaker>     # Align files for <speaker> go here.
data/videos/<speaker>    # Video files for <speaker> go here.
data/features/<speaker>  # Facial feature data for <speaker> is saved here. 
```

## Saving facial feature data
First create the `data/features/<speaker>` directory for the speaker you're
working with. Ensure you also have the align and video files for that speaker,
as outlined above.

Now, all you need to so is run `python pre_processing.py <speaker>` to start
generating and saving speaker data. You can stop at any time by pressing 'q'
(it may take a few seconds to exit). You can pick up where you left off because
the program is smart enough not to reprocess videos it already has data for.

## Results
The paper describing our process and results can be found
[here](https://static.adamheins.com/papers/read-my-lips.pdf).
