# Documentation Resources

Use [terminalizer](https://github.com/faressoft/terminalizer) to record terminal events into animated GIF.
```sh
terminalizer record chatglm -d zsh -k
```

Then render it. This will take a long time. Here I use step size 2 to reduce file size.
```sh
terminalizer render -s 2 chatglm
```

To debug rendering, use a lower quality and larger step. For example:
```sh
terminalizer render -q 50 -s 4 chatglm  # for debugging purpose
```

To further reduce file size, convert the GIF to MP4 format using https://cloudconvert.com/gif-to-mp4 and then convert it back to GIF using https://www.xconvert.com/convert-mp4-to-gif.

Finally, compress the GIF using https://gifcompressor.com/, and that's it.
