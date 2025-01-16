import os
from ultralytics import YOLO
import cv2
from dataset_creation import make_index

base_dir = '/Users/nitthin/Downloads/Invoice-Logo-Detection-main'

images_path = os.path.join(base_dir, '/Users/nitthin/Downloads/Invoice-Logo-Detection-main/images1','test')

model_path = os.path.join('.','/Users/nitthin/Downloads/Invoice-Logo-Detection-main/yolo/runs_colab/runs/detect/train/weights/best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

images = os.listdir(images_path) #path to images

index2logo = {0: 'burgerking',    1: 'worldcup',    2: 'volkswagenautoeps',    3: 'mercedesbenzauto1',    4: 'honda',    5: 'ferrari',    6: 'facebookflat',    7: 'instagram',    8: 'nestleeps',    9: 'redbull',    10: 'fedexoffice',    11: 'nissan',    12: 'jeepnew',    13: 'uber',    14: 'lacoste',    15: 'orange',    16: 'skype',    17: 'lamborghini',    18: 'juventusfc',    19: 'facebookmessenger',    20: 'toshibaeps',    21: 'dell',    22: 'abta',    23: 'uefachampionsleagueeps',    24: 'lg',    25: '2016superbowl',    26: 'blackberry',    27: 'GoDaddy',    28: 'javaeps',    29: 'totals.a01',    30: 'opel',    31: 'androidrobot',    32: 'Canon',    33: 'yandex',    34: 'pumaeps',    35: 'airtel2010',    36: 'kitkat',    37: 'audi',    38: 'appleclassic',    39: 'bookingcom',    40: 'oppoelectronics',    41: 'pinteresticon',    42: 'emiratesairlines01',    43: 'kfcnew',    44: 'Yahoo',    45: 'toyota',    46: 'Instagram',    47: 'underarmourepsai',    48: 'danoneeps',    49: 'amazon',    50: 'newlenovo',    51: 'allahcellacelaluhu',    52: 'acmilaneps',    53: 'chevrolet',    54: 'googlefavicon',    55: 'xiaomi',    56: 'bbnewsfree',    57: 'GoogleDrive',    58: 'YouTubeicon',    59: 'lacosteeps',    60: 'spotify',    61: 'dropbox',    62: 'unicefcyan',    63: 'mcdonald',    64: 'GoogleChrome',    65: 'manchestercity',    66: 'mazda',    67: 'window10',    68: 'nestlefood',    69: 'vansperformance',    70: 'realmadrid',    71: 'nestlechocolat',    72: 'mitsubishi',    73: 'asus',    74: 'nasaeps',    75: 'americanexpress',    76: 'python',    77: 'bmwflat',    78: 'GoogleCalendaricon',    79: 'renault',    80: 'adobeeps',    81: 'vodafoneplc',    82: 'whatsappicon',    83: 'volvocars',    84: 'ajax',    85: 'chicagobulls',    86: 'hpinc',    87: 'ford01',    88: 'unilever01',    89: 'visacard',    90: 'upwork',    91: 'nestlechocolate',    92: 'ebay',    93: 'avgantivirus',    94: 'shell',    95: 'fedexexpress',    96: 'manchesterunited',    97: 'skyeps',    98: 'Ferrari',    99: 'ibm',    100: 'fantaorange',    101: 'nike',    102: 'zte',    103: 'oracle',    104: 'spotify',    105: 'xbox360games',    106: 'jeepblack',    107: 'muhammad',    108: 'bosch',    109: 'Adidas',    110: 'energizer',    111: 'uefaeuropaleague',    112: 'linkedin',    113: 'twitter',    114: 'samsung',    115: 'newfcbarcelona',    116: 'Arsenal',    117: 'fifa',    118: 'liverpool',    119: 'honda3d',    120: 'cocacola',    121: 'mastercard'}

i = 0 #pointer to docs
while i < len(images):    
    img_path = os.path.join(images_path, images[i])
    res = model(img_path)[0]
    bbox = res.boxes.data.tolist()
    img = cv2.imread(img_path)
    # print(res)
    print(bbox)
    if len(bbox) != 0:
        for box in bbox:
            x1, y1, x2, y2, score, class_id = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4) 
            cv2.putText(img, index2logo[int(class_id)], (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
            cv2.imshow(f'image {i+1}',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 
    i += 1
    
