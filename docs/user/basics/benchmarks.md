# Benchmarks

To track our progress over time, we have created a couple of progress trackers, one for our [core functionality](https://concrete.progress.zama.ai) and one for [machine learning](https://ml.progress.zama.ai) built on top of our core funtionality.

We track:
- targets that we want to compile
- status of the compilability of these functions
- evaluation times on different hardwares
- accuracy of the functions when it makes sense
- loss and other metrics of the functions when it makes sense

Note that we are not limited to these, and we'll certainly add more information (e.g., key generation time, encryption time, inference time, decryption time, etc.) once the explicit inference API is available.

Our public benchmarks can be used by competing frameworks or technologies for comparison with **Concrete Numpy**. Notably, you can see:
- if the same functions can be compiled
- what are the discrepancies in the exactness of the evaluations
- how do evaluation times compare

If you want to see more functions in the progress tracker or if there is another metric you would like to track, don't hesitate to drop an email to <hello@zama.ai>.
