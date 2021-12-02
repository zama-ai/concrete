# What is FHE?

Fully Homomorphic Encryption (FHE for short) is a technology that enables computing on encrypted data directly, without having to decrypt it.
Users would encrypt their data using their own secret key, then send it to your servers for processing. Your servers would process the encrypted data blindly, producing a result which itself is encrypted, and that only the user can decrypt with their secret key.

From the user's perspective, nothing changes (but the fact that her data is never in clear on the server), they are still sending data to your service and getting a response. But you now no longer need to worry about securing your user data, as it is now encrypted both in transit and during processing, i.e., it is encrypted end-to-end.

You can learn more about FHE using the following links:
- [quick overview](https://6min.zama.ai/)
- [monthly technical FHE.org meetup](https://www.meetup.com/fhe-org/)
- [videos and resources](http://fhe.org/)

```{warning}
FIXME(Alex/Jeremy): should we link to Zama blogposts or not?
FIXME(Benoit): if yes, I'll do it
```
