# Document Formats
A document file format is a specific way of storing and organizing data within a computer file, designed for documents like text, spreadsheets, presentations, and more. It determines how the information is encoded, allowing software to interpret and display the content correctly.

## Some Common Image Formats
1. PEG (Joint Photographic Experts Group)  
   Features : Lossy compression, good for photographs and continuous-tone images, smaller file sizes.
    Use Cases: Sharing photos online, storing large image collections, web graphics.
2. **PNG** (Portable Network Graphics)  
    Features: Lossless compression, supports transparency, high quality for graphics and illustrations.
    Use Cases: Web graphics with transparency, logos, icons, screenshots.
3. **GIF (Graphics Interchange Format)**  
    Features: Lossless compression, supports animation, limited to 256 colors.
    Use Cases: Simple animations, web graphics with limited color palettes, icons.
4. **TIFF (Tagged Image File Format)**  
    Features: Lossless or lossy compression, high quality, supports various color depths and metadata.
    Use Cases: Professional photography, printing, image editing with preservation of quality.
## Some Common Document Formats
1. **PDF Files**  
    PDF is a portable document format that can be used to present documents that include text, images, multimedia elements, web page links and more.

    ### File Structure  
    - **Header**  
        specifies the version number of the used PDF specification which the document uses. (eg: %PDF-1.4.%.....1)  

    - **Body**  
    In the body of the PDF document, there are objects that typically include text streams, images, other multimedia elements, etc. The Body section is used to hold all the document's data being shown to the user.

    - **xref table**  
    This is the cross reference table, which contains contains the references to all the objects in the document. The purpose of a cross reference table is that it allows random access to objects in the file, so we don't need to read the whole PDF document to locate the particular object. Each object is represented by one entry in the cross reference table, which is always 20 bytes long.

    - **Teailer**  
    The PDF trailer specifies how the application reading the PDF document should find the cross-reference table and other special objects. All PDF readers should start reading a PDF from the end of the file. The last line of the PDF document contains the end of the “%%EOF” file string. Before the end of the file tag, there is a line with a startxref string that specifies the offset from beginning of the file to the cross-reference table.

2. **DOC/DOCX Files**
A Docx file comprises of a collection of XML files that are contained inside a ZIP archive. The contents of a new Word document can be viewed by unzipping its contents. The collection contains a list of XML files that are categorized as:

    MetaData Files - contains information about other files available in the archive
    Document - contains the actual contents of the document  

    DOC is word file saved in Word 2007 and earlier file format. DOCX is based on Open XML file format supported by Microsoft 2007 and later versions.


### Image Processing Libraries in Python
1. OpenCV (Open Source Computer Vision Library)  
    Features: Real-time computer vision, object detection, tracking, feature extraction, image processing, video analysis, machine learning integration.  
    Use Cases: Robotics, augmented reality, facial recognition, medical image analysis, security systems, image preprocessing for ML/DL.

2. Scikit-image
    Features: Image segmentation, color processing, geometric transformations, filtering, feature extraction, analysis, morphology, built on NumPy and SciPy.  
    Use Cases: Scientific image analysis, image classification, object detection, feature engineering for machine learning.

3.  Pillow (Fork of Python Imaging Library - PIL)  
        Features: Image loading, saving, resizing, format conversion, basic manipulations, drawing capabilities.  
        Use Cases: Image manipulation for web development, photo editing, image resizing, format conversion.

4. Scipy  
    Features: Provides the scipy.ndimage submodule specifically designed for multidimensional image processing.  
    Use Cases: Offers functions for linear and non-linear filtering, binary morphology, B-spline interpolation, object measurements, and more.  
    * Complements NumPy with higher-level image processing algorithms, particularly useful for scientific applications.

### Raster vs Vector  
1. Raster
    * Made up of tiny squares called pixels. Each pixel has a specific color, and the arrangement of these pixels forms the image.
    * Dependent on the number of pixels. Higher resolution (more pixels) leads to sharper images, but also larger file sizes.
    * Limited. Enlarging a raster image beyond its original resolution results in pixelation and loss of quality.
    * Well-suited for manipulating photos and images with complex color variations and shading.
    * Common Formats are JPG, PNG, GIF, BMP
    * Use Cases: Photos, digital paintings, web graphics, photo editing.

2. Vector
    *  Defined by mathematical formulas that describe lines, curves, and shapes.
    * Independent of resolution. Can be scaled to any size without losing quality.
    * Infinitely scalable, maintaining sharp edges and smooth lines regardless of size.
    * Easier to edit individual elements (shapes, colors) without affecting the overall image quality.
    * AI, EPS, SVG, PDF (vector)
    * Use Cases: Logos, illustrations, icons, infographics, scalable graphics for print or web.

*Most PDFs are vector files. However, it depends on the program used to create the document because PDFs can also be saved as raster files. For example, any PDF created using Adobe Photoshop will be saved as a raster file.*
