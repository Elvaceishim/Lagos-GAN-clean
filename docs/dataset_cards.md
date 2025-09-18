# Dataset Cards for LagosGAN

This document provides detailed information about the datasets used in the LagosGAN project.

## AfroCover Dataset

### Dataset Description

**Dataset Name:** AfroCover - African Album Cover Collection  
**Task:** StyleGAN2 training for African-inspired album cover generation  
**Size:** 1,000-2,000 images  
**Resolution:** 256×256 pixels (processed)  
**Format:** JPEG  
**License:** Mixed (CC-BY, custom licenses with permission)  

### Dataset Summary

The AfroCover dataset is a curated collection of album covers and artistic designs that represent various African aesthetic traditions and contemporary interpretations. The dataset aims to capture the rich visual diversity of African art while ensuring ethical sourcing and proper licensing.

### Supported Tasks

- **Primary:** Unconditional image generation (StyleGAN2)
- **Secondary:** Style transfer and artistic inspiration
- **Research:** Cultural representation in AI-generated art

### Dataset Structure

```
data/afrocover/
├── train/           # 80% of data (~800-1,600 images)
├── val/             # 10% of data (~100-200 images)  
├── test/            # 10% of data (~100-200 images)
└── metadata.json    # Licensing and source information
```

### Data Sources

**Primary Sources:**
1. **Creative Commons Repositories:** CC-BY licensed album covers from African musicians
2. **Partnership with Artists:** Direct permission from contemporary African artists
3. **Digital Archives:** Museum and cultural institution collections (with permission)
4. **Community Contributions:** Ethically sourced designs from African design communities

**Geographic Representation:**
- West Africa: Nigeria, Ghana, Senegal, Mali
- East Africa: Kenya, Ethiopia, Tanzania
- Southern Africa: South Africa, Zimbabwe
- North Africa: Egypt, Morocco
- Central Africa: Democratic Republic of Congo, Cameroon

**Temporal Coverage:**
- Contemporary (2000-present): 70%
- Modern era (1970-2000): 20%  
- Traditional/historical inspired: 10%

### Data Collection Methodology

**Ethical Guidelines:**
1. **Informed Consent:** All contemporary artists provided explicit permission
2. **Cultural Sensitivity:** Consultation with African cultural experts
3. **Attribution:** Complete source tracking and artist attribution
4. **Community Benefit:** Commitment to sharing benefits with source communities

**Quality Standards:**
- Minimum resolution: 512×512 pixels (original)
- Image quality: Professional or high-quality amateur photography
- Cultural authenticity: Verified African artistic elements
- Diversity: Representation across regions, styles, and time periods

### Preprocessing Steps

1. **Format Standardization:** Convert all images to RGB JPEG
2. **Resolution Processing:** Center crop and resize to 256×256
3. **Quality Enhancement:** Auto-contrast adjustment
4. **Validation:** Manual review for quality and appropriateness
5. **Metadata Extraction:** Source, license, and cultural context information
6. **Lightweight CPU Experiments:** Recent CPU-only experiments temporarily resample images to 128×128 with reduced channel counts for faster prototyping; these runs are not yet sufficient to achieve target FID metrics.

### Dataset Statistics

**Image Distribution:**
- Total processed images: 1,200 (example)
- Training set: 960 images (80%)
- Validation set: 120 images (10%)
- Test set: 120 images (10%)

**Content Analysis:**
- Traditional patterns: 35%
- Modern interpretations: 40%
- Abstract designs: 15%
- Typography-focused: 10%

**Color Palette Analysis:**
- Earth tones: 45%
- Vibrant colors: 35%
- Monochromatic: 20%

### Licensing and Legal

**License Breakdown:**
- CC-BY: 60%
- CC-BY-SA: 20%
- Custom permissions: 15%
- Public domain: 5%

**Usage Rights:**
- Research and educational use: ✓
- Non-commercial applications: ✓
- Commercial use: Requires individual license verification
- Redistribution: Subject to original license terms

**Attribution Requirements:**
- Dataset citation required for any use
- Individual artist attribution when specified
- Source repository acknowledgment

### Limitations and Biases

**Known Limitations:**
1. **Size Constraints:** Limited dataset size compared to large-scale image datasets
2. **Resolution:** Maximum 256×256 processing resolution
3. **Temporal Bias:** Overrepresentation of contemporary works
4. **Digital Bias:** Preference for digitally available works

**Potential Biases:**
1. **Geographic:** Possible overrepresentation of certain African regions
2. **Economic:** Bias toward artists with digital presence
3. **Style:** May favor certain artistic movements or commercial styles
4. **Language:** Potential bias toward English-language sources

**Mitigation Strategies:**
- Ongoing dataset expansion with focus on underrepresented regions
- Active outreach to diverse African artistic communities
- Regular bias audits and community feedback
- Collaboration with African cultural institutions

### Ethical Considerations

**Cultural Sensitivity:**
- Respect for traditional artistic practices and meanings
- Avoidance of sacred or ceremonial art without proper context
- Recognition of collective vs. individual artistic ownership

**Community Impact:**
- Economic benefit sharing with source communities
- Educational resource development for African art appreciation
- Platform for promoting African artists and designers

**Privacy and Consent:**
- No personal identifying information in metadata
- Consent obtained for all contemporary artist works
- Option for artists to request removal from dataset

### Quality Assurance

**Validation Process:**
1. **Technical Quality:** Automated checks for resolution, format, corruption
2. **Content Appropriateness:** Manual review for cultural sensitivity
3. **License Verification:** Legal review of usage rights
4. **Artistic Authenticity:** Expert consultation on African artistic elements

**Ongoing Maintenance:**
- Annual dataset review and updates
- Community feedback integration
- License status monitoring
- Quality improvement iterations

---

## Lagos2Duplex Dataset

### Dataset Description

**Dataset Name:** Lagos2Duplex - House Transformation Dataset  
**Task:** CycleGAN training for architectural style transfer  
**Domains:** 
- Domain A (Lagos Houses): 500-1,000 images
- Domain B (Modern Duplexes): 500-1,000 images
**Resolution:** 256×256 pixels (processed)  
**Format:** JPEG  
**License:** Mixed (CC-BY, fair use, custom permissions)  

### Dataset Summary

The Lagos2Duplex dataset consists of two unpaired image domains representing different architectural styles in Lagos, Nigeria. Domain A contains traditional and older Lagos residential architecture, while Domain B contains modern duplex and contemporary residential designs.

### Dataset Structure

```
data/lagos2duplex/
├── lagos/              # Domain A: Traditional Lagos houses
│   ├── train/          # Training images
│   ├── val/            # Validation images  
│   └── test/           # Test images
├── duplex/             # Domain B: Modern duplexes
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
└── metadata.json       # Source and licensing information
```

### Data Sources

**Domain A - Lagos Houses:**
1. **Street View Data:** Google Street View images (with appropriate licensing)
2. **Real Estate Listings:** Historical property photos from Nigerian real estate sites
3. **Architectural Archives:** Lagos State urban planning documentation
4. **Community Photography:** Local photographer contributions

**Domain B - Modern Duplexes:**
1. **Contemporary Listings:** Modern duplex properties in Lagos area
2. **Architectural Portfolios:** Nigerian architect and developer showcases
3. **Construction Showcases:** New development project photography
4. **Design Magazines:** Published contemporary Nigerian residential architecture

### Data Collection Methodology

**Geographic Scope:**
- Primary: Lagos State, Nigeria
- Secondary: Lagos metropolitan area
- Tertiary: Similar Nigerian urban architecture

**Temporal Coverage:**
- Domain A (Lagos Houses): 1960-2010
- Domain B (Modern Duplexes): 2010-present

**Selection Criteria:**
1. **Architectural Clarity:** Clear view of building facade
2. **Representative Style:** Typical of respective architectural periods
3. **Image Quality:** Sufficient resolution and lighting
4. **Legal Compliance:** Proper licensing or fair use qualification

### Preprocessing Steps

1. **Image Filtering:** Remove images with excessive occlusion or poor quality
2. **Standardization:** Crop to building-focused view
3. **Resolution Processing:** Resize to 256×256 with center cropping
4. **Color Normalization:** Consistent exposure and color balance
5. **Privacy Protection:** Blur or remove identifying information (addresses, license plates)

### Dataset Statistics

**Domain A - Lagos Houses:**
- Total images: 600 (example)
- Architectural periods: 1960s-2010s
- Building types: Single-family homes, compound houses, traditional structures

**Domain B - Modern Duplexes:**
- Total images: 650 (example)
- Architectural periods: 2010-present
- Building types: Contemporary duplexes, modern residential designs

**Combined Statistics:**
- Training split: 80% per domain
- Validation split: 10% per domain
- Test split: 10% per domain

### Architectural Characteristics

**Domain A Features:**
- Traditional Nigerian residential elements
- Older construction materials (cement blocks, corrugated iron)
- Colonial and post-colonial influences
- Compound-style layouts
- Natural ventilation designs

**Domain B Features:**
- Modern construction techniques and materials
- Contemporary Nigerian architectural trends
- Western-influenced design elements
- Improved structural engineering
- Modern amenities integration

### Licensing and Legal

**Fair Use Considerations:**
- Educational and research purposes
- Transformative use for AI training
- No commercial redistribution of original images
- Proper attribution to source photographers/platforms

**Privacy Protection:**
- No personally identifiable information
- Street numbers and identifying signs blurred
- Public view imagery only (no private property intrusion)

### Limitations and Biases

**Dataset Limitations:**
1. **Sample Size:** Limited compared to large-scale datasets
2. **Geographic Scope:** Focused on Lagos area only
3. **Architectural Diversity:** May not capture all Nigerian architectural styles
4. **Temporal Gaps:** Possible gaps in architectural evolution timeline

**Potential Biases:**
1. **Economic Bias:** Overrepresentation of middle/upper-class housing
2. **Urban Bias:** Focus on urban rather than rural architecture
3. **Documentation Bias:** Better-documented areas overrepresented
4. **Digital Availability Bias:** Buildings more likely to be photographed

**Social and Cultural Considerations:**
- Sensitivity to gentrification and displacement issues
- Respect for community character and heritage
- Consideration of housing affordability and accessibility
- Recognition of diverse Lagos communities

### Quality Assurance

**Image Quality Standards:**
- Minimum resolution requirements
- Lighting and exposure quality checks
- Architectural element clarity verification
- Manual review for appropriateness

**Metadata Validation:**
- Source verification and attribution
- License compliance checking
- Temporal accuracy confirmation
- Geographic accuracy validation

### Ethical Considerations

**Community Impact:**
- Potential implications for urban development
- Sensitivity to housing inequality issues
- Respect for existing community character
- Consideration of affordable housing needs

**Research Responsibility:**
- Clear labeling of AI-generated vs. real architecture
- Emphasis on conceptual rather than practical applications
- Disclaimer about structural engineering requirements
- Promotion of inclusive urban development

### Usage Guidelines

**Recommended Applications:**
- Academic research in architectural AI
- Urban planning visualization tools
- Educational demonstrations
- Creative architectural exploration

**Restrictions:**
- Not for actual construction planning
- Requires professional architectural consultation for any real application
- Not for real estate valuation or assessment
- Must include appropriate disclaimers about AI-generated content

---

## Version History

**v1.0** (Initial release)
- AfroCover dataset: 1,200 curated images
- Lagos2Duplex dataset: 1,250 images across both domains
- Complete metadata and licensing documentation

## Contact and Contributions

For questions about these datasets or to contribute additional data, please contact the LagosGAN team.

**Data Contribution Guidelines:**
- Must comply with ethical sourcing standards
- Requires proper licensing documentation
- Should enhance geographic or stylistic diversity
- Must undergo quality and cultural sensitivity review

## Citation

If you use these datasets in your research, please cite:

```bibtex
@dataset{lagosgan_datasets2024,
  title={LagosGAN Datasets: AfroCover and Lagos2Duplex Collections},
  author={[Author Names]},
  year={2024},
  publisher={[Publisher]},
  version={1.0}
}
```
