# Bowties
A python module for modelling bowties in [PyMC](https://www.pymc.io)

## What are Bowties?

[The UK Civil Aviation Authority guide to Bowties](https://www.caa.co.uk/Safety-initiatives-and-resources/Working-with-industry/Bowtie/About-Bowtie/Introduction-to-bowtie/)

## Bowtie Structure
```mermaid
graph LR;
    CA[Cause A]-->B1[Barrier 1];
    CB[Cause B]-->B2[Barrier 2];
    B1[Barrier 1]-->B3[Barrier 3];
    B2[Barrier 2]-->B3[Barrier 3];
    B3[Barrier 3]-->TE[Top Event];
    HC[Hazard Context]-->TE;
    TE-->B5[Barrier 5];
    TE-->B6[Barrier 6];
    B5-->ConA[Consequence A];
    B6-->ConB[Consequence B];
    EF[Escalatory Factor]-->B6;
```

