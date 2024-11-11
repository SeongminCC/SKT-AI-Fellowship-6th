# SKT-AI-Fellowship-6th
This repository is part of the 6th SKT AI Fellowship project on Virtual Try-on. It focuses on enhancing fine-grained garment detail preservation through the use of Edge Filtering and Tunnel Extraction techniques, offering significant improvements in maintaining clothing intricacies during virtual try-on.

## Requirements
```bash
git clone https://github.com/yisol/IDM-VTON.git
cd IDM-VTON

conda env create -f environment.yaml
conda activate idm
```

## Data preparation
#### VITON-HD
You can download VITON-HD dataset from VITON-HD.

After download VITON-HD dataset, move vitonhd_test_tagged.json into the test folder, and move vitonhd_train_tagged.json into the train folder.

Structure of the Dataset directory should be as follows.

```
train
|-- image
|-- image-densepose
|-- agnostic-mask
|-- cloth
|-- vitonhd_train_tagged.json

test
|-- image
|-- image-densepose
|-- agnostic-mask
|-- cloth
|-- vitonhd_test_tagged.json
```

## Train

#### Edge Filtering
```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch train_edge_filtering.py \
--gradient_checkpointing \
--use_8bit_adam \
--checkpointing_epoch=1 \
--num_train_epochs=100 \
--output_dir=train_checkpoints/result_filtering \
--train_batch_size=3 \
--data_dir=/your/path/to/VITON-HD \
--pretrained_model_name_or_path=idm
```
or, you can simply run with the script file.
```bash
sh train_edge_filtering.sh
```

  

#### Tunnel Extraction
```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch train_tunnel_extraction.py \
    --gradient_checkpointing \
    --use_8bit_adam \
    --checkpointing_epoch=1 \
    --num_train_epochs=100 \
    --output_dir=train_checkpoints/result_tunnel \
    --train_batch_size=3 \
    --data_dir=/your/path/to/VITON-HD \
    --pretrained_model_name_or_path=idm
```
or, you can simply run with the script file.
```bash
sh train_tunnel_extraction.sh
```

#### Parameter Descriptions
- **checkpointing_epoch**: The epoch interval at which checkpoints are saved.
- **num_train_epochs**: The total number of training epochs.
- **output_dir**: The directory where training checkpoints will be saved.
- **train_batch_size**: The batch size used for training.
- **data_dir**: The directory containing the training data.
- **pretrained_model_name_or_path**: Path to the pre-trained model weights.



## Inference

```bash
python inference.py \
--base_path=idm_filtere_tunnel \
--device=0 \
--input_dir=/path/to/Input \
--output_dir=/path/to/Output
```
or, you can simply run with the script file.
```bash
sh inference.sh
```

#### Parameter Descriptions
- **base_path**: Path to the pre-trained model weights. Options include idm, idm_filter, or idm_filter_tunnel.
- **device**: GPU number to use for inference.
- **input_dir**: Directory path containing input images.
- **output_dir**: Directory path where output images will be saved.

#### Input Directory Structure
The `input_dir` should be structured as follows:  
```bash
input_dir/
  ├── garment/         # Directory containing clothing images
    ├── garment_1.png
    ├── garment_2.png
  ├── image/           # Directory containing model images
    ├── Lee_1.png
    ├── Lee_2.png
  └── garments.json    # JSON file containing clothing prompts
```

#### Example `garments.json` File
```json
{
    "garment_1.png": "Long Sleeve Polo Collar Shirt",
    "garment_2.png": "Long Sleeve Zip-Up Shirt Collar Sweater"
}
```

#### Constructing Prompts for `garments.json`
Each prompt in the `garments.json` file is constructed by combining values from the following categories in order:  
  
**[Categories]**  
**sleeveLength:**  
```[None, 'Sleeveless', 'Long Sleeve', 'Cropped Sleeve', 'Short Sleeve']```

**neckLine:**  
`['Collarless', 'Bow Collar', 'Shirt Collar', 'Square Neck', 'V Neck', None, 'U Neck', 'Round Neck', 'Stand-up Collar', 'Hood', 'Shawl Collar', 'Offshoulder', 'Tailored Collar', 'Halter Neck', 'Turtle Neck']`

**item:**  
`['flared skirt', 'Tunic Dress', 'Camisole', 'Shirts', 'Pleats Skirt', 'Blouse', 'Full Zip Vest', 'T-shirts', 'Wrap Dress', 'Sarong skirt', 'Wide Pants', 'Leggings/Treggings', 'Sweatshirt', 'Sweater', 'Fur Jacket', 'Fitness Jacket', 'Trumpet Skirt', 'Halter Neck Dress', 'Offshoulder Dress', 'Leather Jacket', 'Polo Shirts', 'Vest', 'Knit vest', 'Bustier', 'Jumper Dress', 'Bolero', 'Bra Top', 'Denim Dress', 'Bikini Top', 'denim skirt', 'Slip Dress', 'pajama-top', 'Knit Dress', 'Vest Suit', 'Wrap Skirt', 'Hoodie', 'Jacket', 'Pique Dress', 'Ruffle skirt', 'Cape/Shawl', 'Shirt Dress', 'Tank Top', 'Sweat Pants', 'Turtleneck', 'Cardigan', 'Y-Shirts', 'Blouson', 'Full Zip Jacket', 'One piece Swimsuit', 'Tube Top']`

**Example prompt construction:**  
``"Long Sleeve Shirt Collar Tunic Dress"``
