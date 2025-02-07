---
title: "LLM Training Strategy"
date: 2018-11-18T12:33:46+10:00
weight: 1
---

Adapting powerful LLM solutions to your business needs through <strong>fine-tuning, reinforcement learning and prompt-engineering</strong>

Language models are making movements in industries but how do we juggle the cost, performance, and scalability? Let’s study how these constraints impact your AI strategy.

---

## TL;DR - Key Business Takeaways

- **Cost vs Performance Trade-off:** Running advanced AI models requires expensive specialized hardware (GPUs), which can either be purchased or rented through cloud services
- **Cloud vs Local Implementation:** Cloud solutions offer flexibility but come with data privacy considerations and availability constraints. Local implementations provide more control but require significant upfront investment
- **Speed Considerations:** Even with high-end hardware, generating AI responses can take time - important to consider for customer-facing applications
- **Quality vs Cost Balance:** While there are ways to reduce costs - like model optimization, they often come with trade-offs in output quality
- **Business Planning Questions (not limited to):** How many users will need to use the AI simultaneously? What response time is acceptable for your use case? How complex does your AI model need to be for your specific needs?
    
    

<aside>
💡 The key is finding the right balance between cost, performance, and quality that aligns with your business needs and budget constraints
</aside>

## Why is the compute cost an issue?

Transformer based language models require enormous levels of compute. To understand what is really going on under the hood, imagine you work at a busy airport with a giant departures board showing flights to hundreds of destinations. Now, instead of listing just the flights, the board also shows how every flight connects to every other flight — layovers and delays. This is typically done by a computer today, but imagine if you had to record and update this information manually. It is clearly a very meticulous task with many dependent moving parts. 

Language models are performing these tasks at a much larger scale. It would be akin to our airport example, but with billions or hundreds of billions of flights (in the literal sense). For example, Meta’s largest [LLaMA 3.1](https://ai.meta.com/blog/meta-llama-3-1/) would be much like an airport scheduler who has to juggle 405 billion flights at any given moment.

To make these calculations, we leverage parallelism, which is like having multiple airport schedulers working together, then combining results into one table. Regular computers that are most widely sold today in electronics shops can handle parallelism, but at a much smaller scale when compared to GPUs. GPUs were originally designed to handle complex calculations quickly through the use of parallelism for commercial gaming or video editing purposes where rendering quality and time is important. Machine learning has been leveraging the parallelism capabilities of the GPU for a while now as well.

Unfortunately, GPUs are expensive, with the highest end commercial grade GPU costing close to $3,000 as of time of writing. Server grade GPUs can run up to tens to hundreds of thousands of dollars to purchase, many of which are manufactured overseas — where prices correlate with political climate. Due to this scarcity, cloud providers who already offer GPU as a service can afford to charge high rates to rent out GPUs for business, academic research, and hobbyist uses.

## What are the most feasible options?

Cloud GPUs offer the benefit of on-demand usage without the need to purchase or maintain the chip. The drawbacks are availability and the requirement of loading sensitive data onto a third party cloud provider’s hardware. This is why local implementations are still desirable. 

When it comes to GPU usage, there are two main considerations. One is how quickly can the GPU make computations, and two is how much data can the GPU handle at any given point in time. For example, the NVIDIA chips A10 typically holds about 24GB of data, while the A100 can hold up to 80GB. Not only do we need to consider the data the chip will process, but also the size of the model being run on the chip. 

### Memory and performance constraints

Language models are a collection of large matrices, each with numbers, or parameters, that take significant memory to store. We calculated, by hand, the number of parameters that OPT-350M requires. This is one of the smaller models that is relatively performant in machine learning literature. (See our calculations in the appendix below. We can also make similar calculations to help you understand your LLM costs.)

By our calculations, [OPT 350m](https://huggingface.co/facebook/opt-350m) requires 356,714,496 parameters. Typically, these are stored as floating point 32 (fp32) numbers, which cost your RAM 4 bytes each. This means we need to use 1,426,857,984 bytes or 1.42 gigabytes just to store the model. This is no problem to store, for even an RTX 4090 which has a RAM of 24 gigabytes.

But how long would it take for an A100 to run through this model? According to NVIDIA specs, the A100 can perform 19.5 trillion operations per second. So a fully loaded context of length 2048 generating 2048 tokens, can be processed in 76.7 seconds. 

So while it seems that on the memory front, the model has no problem being stored in the GPU, to generate 2048 tokens still takes over a minute, even on an A100, which is considered a long time. 

We can improve these numbers through quantization, which has been researched extensively and code to perform quantization exist already. Quantization is simply rounding. Whereas we just analyzed fp32 precision, we can quantize all the parameters down to fp8 or fp4 precision, reducing the memory usage even more, and GPUs perform calculations much faster. 

However, quantization comes at a cost. The current leading theory is that the final digits of the floating point numbers in each parameter of the language model somehow contains the information the language model knows from the ontological standpoint. Thus  with quantization comes degradation in quality of LLM output. That is, there is a three way tradeoff when considering language models and hardware: time, space, and quality. By reducing space, we can improve on time but degrade on quality. By reducing time, e.g. using precomputed weights, we must increase on space and degrade on quality. For the best quality, we need more space, and more time.

## What does this mean for your business?

Aside from understanding how you plan to leverage langauge models, your business should also consider the resource constraints and limitations. How many clients is your language model going to serve at a given moment in time, how tolerant are your clients to latency in the models, how large of a model do you need to serve your business purposes? 

If you can answer those questions, we would be happy to help you consider what hardware you need or should migrate to, and any further optimizations that can be made to increase your client satisfaction. We can also help you determine what resources necessary to best suite your business needs. Book a time for a consult with us and we can begin our roadmap for how LLMs can better serve your business. 

# Appendix

Regarding language models, we can make the following calculations to estimate the memory usage. Let’s take Meta’s OPT-350m - a model with about 350 million parameters and get an even more fine grained estimate on the size this model would take on the GPU. We collect our numbers directly from the [code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/configuration_opt.py). 

1. Every language model begins with its vocabulary - its dictionary of tokens it knows. Here it is 50,272. Each token in the vocabulary is mapped to a high dimensional vector, 512 dimensions. We are at 25,739,264 parameters. We need another one at the output of the model to map embeddings back down to the vocabulary probabilities, so 51,478,528 parameters. 
2. Language models have positional embeddings for each token we query the language model with. In the case of OPT-350m, we have 2050 tokens (extra from 2048 due to padding) that can be entered in at any given time (context size), and the positional embedding vectors are also 1024 dimensional, so we are at 2,099,200 parameters. 
3. The token embeddings are cast up from 512 to 1024 space, to save space. This incurs a cost of 524,288 parameters as opposed to creating another 25 million parameters.
4. The QKV in the attention mechanism is really 3 square matrices of size 1024 by 1024. We also have an output projection matrix, of the same dimensions. These 4 matrices give a dimensionality of 4,194,304. However, we have 16 attention heads and a linear layer of dimension 4096 by 1024, giving a total dimensionality of 4,194,304 + 2 * 4096 * 1024 = 12,582,912 parameters for the transformer block itself. Note that we take advantage of a trick to reduce the parameter count by dividing 1024 by the number of attention heads. This is typical in many transformer implementations. We make 24 successive transformer blocks so there is really 12,582,912 * 24 = 301,989,888 parameters. 
5. We project the 1024 dimensional embeddings back down to 512 dimension, costing another 524,288 parameters. 
6. We skipped a minor step in counting parameters in the attention mechanism. For each transformer block, we have two layernorm steps, one before the feed forward network and one after. Each layernorm costs 2*1024 so in each transformer block, we incur an additional 4096 parameters from the layernorm. Across the 24 layers, we increase our cost by 98,304 parameters, which is not a lot relative to the whole architecture.

In total, our estimate says this model costs 356,714,496 parameters, which is close to the advertised 350 million as in the name OPT-350m. Of course, we do not think the model creators meant the model is exactly 350 million parameters.